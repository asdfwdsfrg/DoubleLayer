from lib.networks import geometry
import torch.nn as nn
import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from . import embedder
import numpy as np
from psbody.mesh import Mesh
import open3d as o3d 

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
 
        self.latent_i = nn.Embedding(cfg.num_train_frame, 128)
        self.latent_c = nn.Embedding(cfg.num_train_frame, 128)

        self.layer_1 = nn.linear(131, 256)
        self.layer_2 = nn.linear(256, 256)
        self.layer_3 = nn.linear(256, 256)
        self.layer_4 = nn.linear(256, 256)
        self.layer_5 = nn.linear(256, 24)
        self.actvn = nn.ReLU()
        
        
        self.feature_fc = nn.Conv1d(256, 256, 1)
        self.latent_fc = nn.Conv1d(384, 256, 1)
        self.view_fc = nn.Conv1d(346, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

    def Normalize(self, blendW_, delta_w):
        base = torch.add(torch.sum(blendW_, dim=1), torch.sum(delta_w, dim=1))
        total = torch.add(blendW_, delta_w) 
        total = torch.transpose(total, 0, 1)
        w = torch.div(total, base).transpose(0, 1)
        return w

    def get_o3d_mesh(vertices, faces):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        return mesh


    def barycentric_interpolation(val, coords):
        """
        :param val: verts x 3 x d input matrix
        :param coords: verts x 3 barycentric weights array
        :return: verts x d weighted matrix
        """
        t = val * coords[..., np.newaxis]
        ret = t.sum(axis=1)
        return ret

    
    def Normalize(weights):
        sum = torch.sum(weights, axis=1)
        weights = torch.transpose(weights, 0, 1)
        normweight = torch.div(weights, sum)
        return torch.transpose(normweight, 0, 1)


    def getcanpts(self, pts, vertices, faces, theta, smpl):
        smpl_mesh = Mesh(vertices, faces)
        closest_face, closest_points = smpl_mesh.closest_faces_and_points(pts) #N * 3, N * 3
        vert_ids, bary_coords = smpl_mesh.barycentric_coordinates_for_points(
        closest_points, closest_face.astype('int32'))
        vert_ids = vert_ids.astype('int32')  #N * 3  
        sh = pts.shape
        delta_w = torch.empty(sh[0], 24)
        for i in range(sh[0]):
            input = torch.cat([pts[i], self.latent_i], 0) 
            net = self.actvn(self.layer_1(input))
            net = self.actvn(self.layer_2(net))
            net = self.actvn(self.layer_3(net))
            net = self.actvn(self.layer_4(net))
            output = self.layer_5(net)
            delta_w[i] = output 
        delta_w = self.Normalize(delta_w)  #pts * 24
        blendW = smpl.blendW[vert_ids, :]  #3 * 24
        bweights = self.barycentric_interpolation(blendW, bary_coords)
        weights = smpl.Normalize(delta_w, bweights)
        canpts = smpl.Getlbs(theta, pts=pts, weights=weights, inverse=True)
        return canpts
        
    # def pts_to_can_pts(self, pts, sp_input):
    #     """transform pts from the world coordinate to the smpl coordinate"""
    #     Th = sp_input['Th']
    #     pts = pts - Th
    #     R = sp_input['R']
    #     pts = torch.matmul(pts, R)
    #     return pts

    def calculate_density(self, wpts, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)
        grid_coords = self.get_grid_coords(ppts, sp_input)
        grid_coords = grid_coords[:, None, None]
        xyzc_features = self.interpolate_features(grid_coords, feature_volume)

        # calculate density
        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)
        alpha = alpha.transpose(1, 2)

        return alpha

    def calculate_density_color(self, wpts, viewdir, feature_volume, sp_input):
        # interpolate features
        # ppts = self.pts_to_can_pts(wpts, sp_input)
        # grid_coords = self.get_grid_coords(ppts, sp_input)
        # grid_coords = grid_coords[:, None, None]
        # xyzc_features = self.interpolate_features(grid_coords, feature_volume)

        # calculate density
        net = self.actvn(self.fc_0(wpts))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)

        # calculate color
        features = self.feature_fc(net)

        latent = self.latent(sp_input['latent_index'])
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        light_pts = embedder.xyz_embedder(wpts)
        light_pts = light_pts.transpose(1, 2)

        features = torch.cat((features, viewdir, light_pts), dim=1)

        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw

    def forward(self, sp_input, grid_coords, viewdir, light_pts):
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        p_features = grid_coords.transpose(1, 2)
        grid_coords = grid_coords[:, None, None]

        code = self.c(torch.arange(0, 6890).to(p_features.device))
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)

        xyzc_features = self.xyzc_net(xyzc, grid_coords)

        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.latent(sp_input['latent_index'])
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = viewdir.transpose(1, 2)
        light_pts = light_pts.transpose(1, 2)
        features = torch.cat((features, viewdir, light_pts), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw
