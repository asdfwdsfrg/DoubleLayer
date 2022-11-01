import torch
from lib.config import cfg
from lib.networks.body_model import BodyModel
from .nerf_net_utils import *
from line_profiler import LineProfiler
from . import embedder
import os
import psutil

class Renderer:
    def __init__(self, net, body):
        self.net = net
        self.body = body

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def prepare_input(self, batch):
        sp_input = {}

        sh = batch['rgb'].shape
        # idx = [torch.full([sh[1]], i) for i in range(sh[0])] 
        # idx = torch.cat(idx).to(batch['coord'])
        # coord = batch['coord'].view(-1, sh[-1])
        # sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)
        sp_input['batch_size'] = sh[0]
        sp_input['bounds'] = batch['bounds']
        sp_input['R'] = batch['R']
        sp_input['Th'] = batch['Th']
        sp_input['params'] = batch['params']
        sp_input['latent_index'] = batch['latent_index']

        return sp_input

    def get_density_color(self, wpts, viewdir, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        raw = raw_decoder(wpts, viewdir)
        return raw

    def get_pixel_value(self, ray_o, ray_d, near, far, wpts, z_vals, raw, 
                        sp_input, batch):
        # viewing direction 
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        z_vals = z_vals.view(-1, n_sample)
        #n_rays x N_sample x 4 以射线为最小单位渲染
        raw = raw.reshape(-1, n_sample, 4)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        ret = {
            'rgb_map': rgb_map.view(n_batch, n_pixel, -1),
            'disp_map': disp_map.view(n_batch, n_pixel),
            'acc_map': acc_map.view(n_batch, n_pixel),
            'weights': weights.view(n_batch, n_pixel, -1),
            'depth_map': depth_map.view(n_batch, n_pixel)
        }

        return ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        params = batch['params']
        sh = ray_o.shape

        
        input = self.prepare_input(batch)
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)
        #batchs X R_ray X N_sampled X 3
        #calculate color & density
        # lp = LineProfiler()
        
        # lp_wrapper = lp(self.net.forward) 
        # c, d, ei, nodes_delta, mean, std = lp_wrapper(input, wpts, self.body)
        c, d, ei, nodes_delta, mean, logvar= self.net(input, wpts, self.body)
        
        # lp.print_stats()
        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        c = c.view(-1, n_pixel, cfg.N_samples, 3)
        d = d.view(-1, n_pixel, cfg.N_samples, 1)
        # Batchs X R_ray X N_samples X 3
        raw = torch.cat([c, d], dim = -1)
        
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, wpts, z_vals,
                                               raw, input, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}
        ret['delta_nodes'] = nodes_delta.transpose(0, 1)
        ret['embedding'] = ei.transpose(0,1)
        ret['mean'] = mean.transpose(0, 1)
        ret['logvar'] = logvar.transpose(0, 1)
        return ret
