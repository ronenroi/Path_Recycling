import os, sys

import numpy as np

my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene import *
from classes.scene_rr import *
from classes.camera import *
from classes.visual import *
from utils import *
from cuda_utils import *
import matplotlib.pyplot as plt
from classes.tensorboard_wrapper import TensorBoardWrapper
import pickle
from classes.checkpoint_wrapper import CheckpointWrapper
from time import time
from classes.optimizer import *
from os.path import join
import glob
from tqdm import tqdm
import json
from scipy.sparse import csr_matrix
cuda.select_device(0)
from scipy.io import loadmat
#############################
# Simulation basic unit = km #
#############################

########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
data_dir = "/home/roironen/Path_Recycling/code/projects/cloud_rendering_vadim"
files = glob.glob(join(data_dir,"*.mat"))
N_cloud = 1

beta_cloud = loadmat(files[0])['beta'] * 1000 # 1/m -> 1/km
beta_cloud = beta_cloud.astype(float_reg)

# Grid parameters #
# bounding box
voxel_size_x = 0.02 #20m
voxel_size_y = 0.02 #20m
voxel_size_z = 0.02 #20m
edge_x = voxel_size_x * beta_cloud.shape[0]
edge_y = voxel_size_y * beta_cloud.shape[1]
edge_z = voxel_size_z * beta_cloud.shape[2]
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])


print(beta_cloud.shape)
print(bbox)

beta_air = 0.004
w0_air = 0.912
w0_cloud = 0.99
g_cloud = 0.85


#######################
# Cameras declaration #
#######################
# height_factor = 2
fov = np.deg2rad(1)
ps_max = 512

sensor_size = 5e-9 * ps_max #pixel width in km * number of pixels
#   #np.array((50e-3, 50e-3)) / height_factor

focal_length = sensor_size / (2 * np.tan(fov / 2))#10e-3
sensor_size = np.array((sensor_size, sensor_size))

pixels = np.array((ps_max, ps_max))

N_cams = 1
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
R = 600

t = R * theta_phi_to_direction(0,0) + volume_center
euler_angles = np.array((180, 0, -90))
cameras.append(Camera(t, euler_angles, focal_length, sensor_size, pixels))




# Simulation parameters
N_render_gt = 10
Np_gt = int(1e8)
# Np_gt *= N_render_gt



resample_freq = 1
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.05
# to_mask = True

beta_max = beta_cloud.max()

print(beta_cloud.mean())
print(beta_cloud.max())

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
scene_rr = LargeDomainSceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
visual = Visual_wrapper(scene_rr)
# visual.create_grid()
# visual.plot_cameras()
# visual.plot_medium()
plt.show()


for i in range(N_render_gt):
    cuda_paths = scene_rr.build_paths_list(Np_gt)
    print(i)
    if i == 0:
        I_gt = scene_rr.render(cuda_paths)
    else:
        I_gt += scene_rr.render(cuda_paths)
I_gt /= N_render_gt

del(cuda_paths)
cuda_paths = None
max_val = np.max(I_gt, axis=(1,2))
visual.plot_images(I_gt/max_val, "GT")
plt.show()
