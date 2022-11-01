import os
import sys

import numpy as np
import torch
sys.path.append("../")
from lib.networks.body_model import SMPLlayer
from lib.visualizers.vis_3d import vis_3d

param_dir = 'data/zju_mocap/CoreView_313/new_params'
verts_dir = 'data/zju_mocap/CoreView_313/new_vertices'

# Previously, EasyMocap estimated SMPL parameters without pose blend shapes.
# The newlly fitted SMPL parameters consider pose blend shapes.
new_params = False
if 'new' in os.path.basename(param_dir):
    new_params = True

param_path = os.path.join(param_dir, "1178.npy")
verts_path = os.path.join(verts_dir, "1178.npy")

## load precomputed vertices
verts_load = np.load(verts_path) #6890*3
#vis_3d(verts_load)

## create smpl model
model_folder = 'smpl_model'
device = torch.device('cpu')
body_model = SMPLlayer(os.path.join(model_folder),
                       gender='female',
                       device=device,
                       #regressor_path=os.path.join(model_folder,'J_regressor_body25.npy')\
                       )
body_model.to(device)

## load SMPL zju
params = np.load(param_path, allow_pickle=True).item()

vertices = body_model(return_verts=True,
                      return_tensor=False,
                      new_params=new_params,
                      **params)
vis_3d(vertices)


