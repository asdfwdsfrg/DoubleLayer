import os
from random import randrange
from re import I
from threading import local
from xmlrpc.client import boolean

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.config import cfg
from lib.networks.body_model import BodyModel
from lib.networks.embedder import *
from lib.utils.write_ply import write_ply

from . import embedder


class Network(nn.Module):
    def __init__(self):

        super(Network, self).__init__()

         
        #encoder
        self.ec_ly1 = nn.Conv1d((72 + time_dim) * 128, 64 * 128, kernel_size= 1, groups =128)
        self.ec_ly21= nn.Conv1d(64 * 128, 8 * 128, kernel_size=1, groups=128)
        self.ec_ly22= nn.Conv1d(64 * 128, 8 * 128, kernel_size=1, groups=128)
       
        
        
        #decoder
        self.dc_ly1 = nn.Conv1d(80 * 128, 64 * 128, kernel_size= 1, groups =128)
        self.dc_ly21= nn.Conv1d(64 * 128, 32 * 128, kernel_size=1, groups=128)
        self.dc_ly22= nn.Conv1d(64 * 128, 3 * 128, kernel_size=1, groups=128)
       
        self.actvn = nn.ReLU()
        self.actvn2 = nn.Sigmoid()  
        # nodes x s' x (32+xyz_dim)

        #feature field
        self.f_ly1 = nn.Conv1d((32 + xyz_dim) * 128, 64 * 128, kernel_size=1, groups = 128)
        self.f_ly2 = nn.Conv1d(64 * 128, 64 * 128, kernel_size=1, groups = 128)
        self.f_ly3 = nn.Conv1d(64 * 128, 64 * 128, kernel_size=1, groups = 128)
        self.f_ly4 = nn.Conv1d(64 * 128, 256 * 128, kernel_size=1, groups = 128)

        #Nerf color
        self.c_ly1 = nn.Linear(256, 64)
        self.c_ly2 = nn.Linear(64, 3)

        #NerF density
        self.d_ly1 = nn.Linear(256, 1)

    def encode(self, t_ped, poses, w):
        '''
        poses: batch x nodes_n x 24 x 3
        t_ped: batch x time_dim
        mean, std:  Batchs x nodes x 8
        w: attention_map : 128 x 24
        '''
        batch_size = poses.shape[0]
        nodes_n = cfg.n_nodes
        #B x N_nodes x 24 X 3
        #B x N_nodes x 72
        poses = torch.mul(w, poses).view(batch_size,nodes_n,72)
        #B x N_nodes x time_dim
        t_ped = t_ped.expand(nodes_n, batch_size, time_dim).permute(1, 0, 2)
        #B x N_nodes X (72 + time_dim)
        encoder_in = torch.cat([poses, t_ped], dim = -1).permute(1, 2, 0).reshape(nodes_n * (72 + time_dim), batch_size)
        #nodes x 64 x batchs
        net = self.actvn(self.ec_ly1(encoder_in))
        #nodes x 8 x batchs
        mean = self.ec_ly21(net).view(nodes_n, 8, batch_size).permute(0,2,1)
        logvar = self.ec_ly22(net).view(nodes_n, 8, batch_size).permute(0, 2, 1)
        return mean, logvar
    
    def decode(self, z, poses, w):
        '''
        poses: Batchs x 1 x 72
        z: nodes x Batchs x 8
        ei: Batchs x nodes x 32
        '''
        batch_size = poses.shape[0]
        nodes_n = cfg.n_nodes
        #B x N_nodes x 72
        poses = torch.mul(w, poses).view(batch_size,nodes_n,72)
        #N_nodes x 80 x batchs
        input = torch.cat([poses, z.transpose(0, 1)], dim = -1).permute(1, 2, 0).reshape(nodes_n * 80, batch_size)
        #N_nodes x 64 x batchs
        net = self.actvn(self.dc_ly1(input))
        ei = self.dc_ly21(net).view(nodes_n, 32, batch_size).permute(0, 2, 1)
        delta_ni = self.dc_ly22(net).view(nodes_n, 3, batch_size).permute(0, 2, 1)
        #nodes x batches x (32, 3)
        return ei, delta_ni


    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        else:
            return mean
    
    def Feature_field(self, ei, local_coords, mask, mark, s, s_max):
        '''
        local_coords: nodes x B x V x 3
        ei:nodes x B x 32
        s: nodes x B 
        return 
            f:nodes x s_max x 256
        '''
        nodes_n = ei.shape[0]
        batch_size = ei.shape[1] 
        ei = ei.expand(local_coords.shape[-2], nodes_n, batch_size, ei.shape[-1])
        ei = ei.permute(1, 2, 0, 3)
        # V x (32 + xyz_dim + 1)
        raw = torch.cat([ei, local_coords], dim = -1)[mask]
        if(s_max == 0):
            return torch.zeros(nodes_n, s_max, 256)
        #nodes_n X S' X (32 + xyz_dim)
        input = torch.zeros([local_coords.shape[0], s_max, 32 + xyz_dim], device = ei.device) #type: ignore
        for i in range(local_coords.shape[0]): 
            input[i][:s[i]] = raw[mark == i]
        input = input.permute(0,2,1).reshape(nodes_n * (32 + xyz_dim), s_max)
        # nodes x (32 + xyz_dim) x S_max
        net = self.actvn(self.f_ly1(input))
        net = self.actvn(self.f_ly2(net))
        net = self.actvn(self.f_ly3(net))
        output = self.actvn(self.f_ly4(net))
        f = output.view(nodes_n, 256 , s_max).permute(0, 2, 1)
        return f



    def blend_feature(self, bweights, f):
        '''
        bweights: nodes x B x V
        f: nodes x B x V x 256
        '''
        bweights = bweights.permute(1, 2, 0)
        bweights += 0.0000001
        weights_sum = torch.sum(bweights, dim = -1)
        bweights = bweights / weights_sum[...,None]
        f = f.permute(1, 2, 3, 0)
        #B x V x 256
        f_blend = torch.matmul(f, bweights[...,None]).squeeze(dim = -1)
        return f_blend 

    def Nerf(self, f):
        '''
        f: B x V x 256
        '''
        netc = self.actvn(self.c_ly1(f))
        c = self.c_ly2(netc)
        d = self.actvn(self.d_ly1(f))
        return c, d

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def blend_weights(self, wpts, nodes_posed):
        """feature blending weights"""    
        '''
            wpts:B x V x 3
            nodes_posed: B x nodes_n x 3
            return:
                bweights: nodes_n X B X V  the nodes influence on each vertex
                mask: nodes x B X V  bool  knb[i][j] = 1 means Vj inside the nodes_i's influence range
        '''
        mask = torch.zeros([nodes_posed.shape[1], wpts.shape[0], wpts.shape[1]], dtype=boolean, device = wpts.device)
        bweights = torch.zeros([nodes_posed.shape[1], wpts.shape[0], wpts.shape[1]], device = wpts.device)
        for i in range(nodes_posed.shape[1]):
            norm_2 = torch.sum(torch.pow(wpts - nodes_posed[:, i, :].unsqueeze(dim = 1), 2), dim = -1)
            nodes_influ = torch.exp(- norm_2 / (2 * torch.pow(torch.tensor(cfg.sigma), 2))) - torch.tensor(cfg.epsilon)
            mask[i] = nodes_influ.ge(0)
            bweights[i] = torch.max(nodes_influ, torch.tensor(0).float())
        return bweights, mask

    def calculate_local_coords(self, wpts, nodes_T, nodes_weights, J, body, poses, shapes, R, Th):
        """get local_coords"""
        '''
            pts: nodes_n x B x V x  3
            nodes: B x nodes_n x 3
            nodes_delta: nodes_n x B x 3
            nodes_weights: nodes_n x Joints
            mask: nodes_n x B x V
            s: nodes_n x B
            return:
                local_coords: nodes_n x B x V x 4
        '''
        #V x B x nodes_n x 3
        nodes_n = cfg.n_nodes
        wpts = wpts.expand(nodes_n, wpts.shape[0], wpts.shape[1], wpts.shape[2])
        wpts = wpts.permute(2,1,0,3)
        coords_T, j_transformed = body.get_lbs(poses, shapes, nodes_weights, wpts, joints = J, Rh = R, Th = Th, inverse = True)
        #V x B x nodes_n x 3
        local_coords = coords_T - nodes_T 
        return local_coords.permute(2, 1, 0, 3)

    def forward(self, input, wpts, body):
        #transform sampled points from world coord to smpl coord
        batch_size = input['batch_size']
        nodes_n = cfg.n_nodes
        #B x V x 3
        weights = body.basis['weights']
        nodes_ind = body.basis['nodes_ind']
        nodes_weights= weights[nodes_ind]
        wpts = wpts.view(batch_size, -1, 3)
        wpts = self.pts_to_can_pts(wpts, input)
        params = input['params']
        poses = params['poses'].squeeze(dim = -2)
        poses_exp = poses.expand(nodes_n, batch_size, 72).permute(1, 0, 2).reshape(batch_size, nodes_n, 24, 3)
        w = torch.matmul(body.attention_map, nodes_weights[..., None]).squeeze(dim=-1)
        w = w.expand(batch_size, 3, 128, 24).permute(0, 2, 3, 1)
        shapes = params['shapes']
        R = input['R']
        Th = input['Th']
        #nodes in T pose
        nodes_T = body.basis['v_shaped'][nodes_ind]
        batch_nodes_T = nodes_T.expand(batch_size, nodes_T.shape[-2], nodes_T.shape[-1])
        t_ped = time_embedder(input['latent_index']).view(batch_size,-1)
        mean, logvar = self.encode(t_ped, poses_exp, w)
        z = self.reparameterize(mean, logvar)
        #nodes x B x (32, 3)
        ei, nodes_delta = self.decode(z, poses_exp, w)  
        batch_nodes_T = batch_nodes_T + nodes_delta.transpose(0, 1) 

        batch_nodes_posed, j_transformed = body.get_lbs(poses, shapes, nodes_weights, batch_nodes_T, Rh = R, Th = Th)
        #B X V X 3
        #bweights: nodes_n X B X V  the nodes influence on each vertex
        #mask: nodes_n x B x V
        bweights, mask = self.blend_weights(wpts, batch_nodes_posed)
        pts_nodes_ind = torch.range(0, 127, dtype=int,device=wpts.device)
        pts_nodes_ind = pts_nodes_ind.expand(batch_size, wpts.shape[1], cfg.n_nodes).permute(2, 0, 1)[mask]
        #s[i]: the number of pts influenced by nodes_i
        f = torch.zeros(nodes_n, batch_size, wpts.shape[-2], 256, device = wpts.device)
        s = torch.sum(torch.sum(mask, dim = 2), dim = 1)
        s_max = torch.max(s)
        #s: nodes_n x B the number of pts under the nodes' influence
        if s_max == 0:
            f_blend = self.blend_feature(bweights, f)
            c, d = self.Nerf(f_blend)
            return c, d, ei, nodes_delta, mean, logvar
      
        #nodes_n x B x V x 3
        local_coords = self.calculate_local_coords(wpts, batch_nodes_T, nodes_weights, j_transformed, body, poses, shapes, R, Th) #TODO High complexity
        #n x b x v x 256
         #
        f_hit = self.Feature_field(ei, xyz_embedder(local_coords), mask, pts_nodes_ind, s, s_max) #TODO high complexity 75%
         
        mask_f = torch.zeros(f_hit.shape[0], f_hit.shape[1], dtype=bool, device = f.device)
        for i, pts_num in enumerate(s):
            mask_f[i][:pts_num] = True
        f_hit = f_hit[mask_f]
        f[mask] = f_hit

        f_blend = self.blend_feature(bweights, f)
        #batchs x v x 3,1
        c, d = self.Nerf(f_blend)
        #ei, nodes_delta: nodes x batches x (32, 3)
        #mean, std: nodes x batchs x 8 
        return c, d, ei, nodes_delta, mean, logvar

