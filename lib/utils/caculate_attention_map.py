import numpy as np

def cal_attention_map(p):
    distance = np.zeros([24,24], dtype = np.float32)
    for i, j in enumerate(p):
        distance[i][j] = 1
        distance[j][i] = 1
    d2 = np.matmul(distance, distance)
    d3 = np.matmul(d2, distance)
    d4 = np.matmul(d3, distance)
    d4[d4 != 0] = 1
    return d4
