import torch
import os
import torch.nn as nn

from .lbs import blend_shapes, vertices2joints
from .lbs import batch_rodrigues, lbs 
import os.path as osp
import pickle
import numpy as np
from lib.utils.caculate_attention_map import cal_attention_map


def to_tensor(array, dtype=torch.float32, device=torch.device('cpu')):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype).to(device)
    else:
        return array.to(device)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class BodyModel(object):
    def __init__(self,
                 model_path,
                 n_nodes,
                 betas,
                 gender='neutral',
                 device=None) -> None:
        dtype = torch.float32
        self.dtype = dtype
        self.device = device
        # create the SMPL model
        if osp.isdir(model_path):
            model_fn = 'SMPL_{}.{ext}'.format(gender.upper(), ext='pkl')
            smpl_path = osp.join(model_path, model_fn)
        else:
            smpl_path = model_path
        assert osp.exists(smpl_path), 'Path {} does not exist!'.format(
            smpl_path)

        with open(smpl_path, 'rb') as smpl_file:
            data = pickle.load(smpl_file, encoding='latin1')
        self.faces = data['f']
        self.basis = {}
        self.basis['face_tensor'] = to_tensor(to_np(self.faces, dtype=np.int64), dtype=torch.long)
        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*3 x 207
        num_pose_basis = data['posedirs'].shape[-1]
        # 207 x 20670
        posedirs = data['posedirs']
        data['posedirs'] = np.reshape(data['posedirs'], [-1, num_pose_basis]).T

        for key in [
                'J_regressor', 'v_template', 'weights', 'posedirs', 'shapedirs', 'J'
        ]:
            val = to_tensor(to_np(data[key]), dtype=dtype)
            self.basis[key] = val
        # indices of parents for each joints
        self.basis['parents'] = to_tensor(to_np(data['kintree_table'][0])).long()
        self.basis['parents'][0] = 0
        #Sample Nodes
        v_shaped = self.basis['v_template'] + blend_shapes(betas,self.basis['shapedirs'])
        self.basis['v_shaped'] = v_shaped.squeeze()
        # self.farest_point_sampling(n_nodes, self.basis['v_shaped'])
        self.attention_map = torch.from_numpy(cal_attention_map(self.basis['parents'])).to(self.device)
        nodes_path = os.path.join('nodes_T.npy')
        self.basis['nodes_ind'] = torch.from_numpy(np.sort(np.load(nodes_path)))
        self.basis['J_shaped'] = vertices2joints(self.basis['J_regressor'], v_shaped)
        for key in self.basis:
            self.basis[key] = self.basis[key].to(self.device)

    def farest_point_sampling(self, n_samples, v_shaped):
        #6890 x 6890 
        distance = torch.stack([torch.sqrt(torch.sum(torch.pow(v_shaped - v_shaped[i], 2),dim = 1)) for i in range(v_shaped.shape[0])], dim = 0)
        n_sampled = [0]
        v_candidate = [i for i in range(1, v_shaped.shape[0])]
        for i in range(n_samples - 1):
            far_p = 0
            far_p_ind = 0
            dis = 0
            for j in v_candidate:
                dis = -1
                for k in n_sampled:
                    if dis > distance[j][k] or dis == -1: 
                        dis = distance[j][k]
                if far_p < dis:     
                    far_p = dis
                    far_p_ind = j
            n_sampled.append(far_p_ind)
            v_candidate.remove(far_p_ind)
        np.save('nodes_T.npy', n_sampled)
        return n_sampled
        

    def get_lbs(self,
                poses,
                shapes,
                weights,
                pts,
                joints = None,
                Rh=None,
                Th=None,
                return_verts=True,
                return_tensor=True,
                scale=1,
                new_params=False,
                inverse = False):
        """Do LBS or Inverse LBS 

        Args:
            poses (B, 72)
            shapes (B, 10)
            weigths (B, V, 24)
            pts(... x B x V x 3)
            joints(B x 24 x 3) : default the shaped T pose joints
            Rh (B, 3): global orientation
            Th (B, 3): global translation
            return_verts (bool, optional): if True return (6890, 3). Defaults to False.
        """
        if 'torch' not in str(type(poses)):
            dtype, device = self.dtype, self.device
            poses = to_tensor(poses, dtype, device)
            shapes = to_tensor(shapes, dtype, device)
            Rh = to_tensor(Rh, dtype, device)
            Th = to_tensor(Th, dtype, device)
        bn = poses.shape[0]
        rot = Rh
        if Rh is None:
            Rh = torch.zeros(bn, 3, device=poses.device)
            rot = batch_rodrigues(Rh)
        if Th is None:
            Th = torch.zeros(bn, 3, device = poses.device)
            Th = Th.unsqueeze(dim = 0)
        transl = Th.unsqueeze(dim=1)
        if shapes.shape[0] < bn:
            shapes = shapes.expand(bn, -1)
        if joints is None:
            joints = self.basis['J_shaped'].expand(bn, 24, 3)
        # if inverse:
        #     print(rot.shape, pts.shape)
        #     pts = torch.matmul(rot.transpose(1,2) * scale, pts - transl)
        #     joints = torch.matmul(rot.transpose(1,2) * scale, joints - transl)
        # if return_verts:
        vertices, j_transformed = lbs(shapes,
                                poses,
                                pts,
                                self.basis['shapedirs'],
                                self.basis['posedirs'],
                                joints,
                                self.basis['parents'],
                                weights,
                                pose2rot=True,
                                new_params=new_params,
                                dtype=self.dtype,
                                inverse = inverse)
        # vertices = (torch.matmul(vertices, rot.transpose(1, 2)) * scale + transl)[0]
        # j_transformed = (torch.matmul(j_transformed, rot.transpose(1, 2)) * scale + transl)[0]
        # if not return_tensor:
            # vertices = vertices.detach().cpu().numpy()
            # transl = transl.detach().cpu().numpy()
        return vertices, j_transformed
