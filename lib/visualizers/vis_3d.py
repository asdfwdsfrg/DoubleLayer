from platform import node
import cv2 
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

def vis_3d_heat(v, mask, bweights, nodes_posed):
#v : B x V x 3
    nodes_posed = nodes_posed.detach().cpu().numpy()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    x = nodes_posed[0, :, 0]
    y = nodes_posed[0, :, 1]
    z = nodes_posed[0, :, 2]
    color = (x + y + z)
    colormap = cm.get_cmap('hsv')
    color = colormap(color)
    ax.scatter(x, y, z, s = 20, c = color[:,:3])
    plt.show()

