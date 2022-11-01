import copy
import datetime
import os
import time

import numpy as np
import torch
import tqdm
from line_profiler import LineProfiler
from torch.distributions import Normal, kl_divergence
from torch.nn import DataParallel

from lib.config import cfg
from lib.networks import nerf_renderer
from lib.networks.body_model import BodyModel


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        network = network.to(device)
        if cfg.distributed:
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank
            )
        self.network = network
        self.local_rank = cfg.local_rank
        self.device = device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def batch_slice(self, batch, from_i, mini_batch_size):
        minibatch = copy.deepcopy(batch)
        for k in batch:
            if k in ['ray_o', 'ray_d']:
                minibatch[k] = batch[k][:,from_i : from_i + mini_batch_size, :]
            if k in ['near', 'far']:
                minibatch[k] = batch[k][:,from_i : from_i + mini_batch_size]
        return minibatch        

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            if isinstance(batch[k], dict):
                for key in batch[k]:
                    batch[k][key] = batch[k][key].to(self.device)
            else:
                batch[k] = batch[k].to(self.device)
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()

        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1

            batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats= self.network(batch)
            # training stage: loss; optimizer; scheduler
            optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            #gradient clip
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

            if iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train', image_stats = image_stats) 

    def calculate_loss(self, ret, batch):
        img2mse = lambda x, y : torch.sum(torch.sum((x - y) ** 2, dim = 1),dim=0)
        # mask = batch['msk']
        # H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        # mask = mask.reshape(H, W)
        # img_pred = np.zeros((H, W, 3))
        # img_gt = np.zeros((H, W, 3))
        
        mask = batch['msk']
        scalar_stats = {}
        loss = 0
        lambda_img_loss = cfg.lambda_img_loss
        lambda_trans = cfg.lambda_trans 
        lambda_ebd = cfg.lambda_ebd
        lambda_kl = cfg.lambda_kl
        img_loss = img2mse(ret['rgb_map'], batch['rgb'])
        scalar_stats.update({'img_loss': img_loss})
        # normal_pred = Normal(ret['mean'], ret['std'])
        # normal_t = Normal(0, 1)
        # kl_loss = kl_divergence(normal_pred, normal_t)
        # kl_loss = torch.sum(kl_loss.view(-1, 1), dim = 0)
        kl_loss = -0.5 * torch.sum(1 + ret['logvar'] - ret['mean'].pow(2) - ret['logvar'].exp())
        trans_loss = torch.norm(ret['delta_nodes']) ** 2
        ebd_loss = torch.norm(ret['embedding']) ** 2
        loss += lambda_img_loss * img_loss + lambda_ebd * ebd_loss + lambda_kl * kl_loss + lambda_trans * trans_loss
        scalar_stats.update({'loss': loss})
        image_stats = {}
        image = ret['rgb_map']
        image_stats.update({'image':image})
        return ret, loss, scalar_stats, image_stats


    def val(self, epoch, data_loader, network, evaluator=None, recorder=None):
        network.eval()
        torch.cuda.empty_cache()
        params_path = os.path.join('data/zju_mocap/CoreView_387/new_params/0.npy')
        #init structure model
        betas = np.load(params_path, allow_pickle=True).item()['shapes']
        betas = torch.from_numpy(betas)
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        body = BodyModel(os.path.join('smpl_model'), 128, betas, gender = 'male', device = device)
        renderer = nerf_renderer.Renderer(network, body)        
        val_loss_stats = {}
        data_size = len(data_loader)
        j = 0
        for batch in tqdm.tqdm(data_loader):
            j = j + 1
            o = {}
            o['rgb_map'] = list()
            o['mean'] = list()
            o['logvar'] = list()
            o['embedding'] = list()
            o['delta_nodes'] = list()
            output = {}
            with torch.no_grad():
                batch = self.to_cuda(batch)
                r_size = batch['ray_o'].shape[1]
                for i in range(0, r_size, 2048):
                    minibatch = self.batch_slice(batch, i, 2048)
                    # lp = LineProfiler()
                    # lp.add_function(network.forward)
                    # lp_wrapper = lp(renderer.render)
                    o_i = renderer.render(minibatch)
                    # o_i = lp_wrapper(minibatch)
                    # lp.print_stats()
                    o['rgb_map'].append(o_i['rgb_map'])
                    o['mean'].append(o_i['mean'])
                    o['logvar'].append(o_i['logvar'])
                    o['embedding'].append(o_i['embedding'])
                    o['delta_nodes'].append(o_i['delta_nodes'])
            output['rgb_map'] = torch.cat(o['rgb_map'], dim = 1)
            output['mean'] = torch.cat(o['mean'], dim = 1)
            output['logvar'] = torch.cat(o['logvar'], dim = 1)
            output['embedding'] = torch.cat(o['embedding'], dim = 1)
            output['delta_nodes'] = torch.cat(o['delta_nodes'], dim = 1)
            ret, loss, loss_stats, image_stats = self.calculate_loss(output, batch) 
            if evaluator is not None:
                evaluator.evaluate(output, batch)
            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)
# 
        if evaluator is not None:
            result = evaluator.summarize()
            # val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
