import shutil
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
import scipy
cuda.select_device(0)

sun_angles = np.array([180, 0]) * (np.pi / 180)
data_dir = "/home/roironen/pytorch3d/projects/CT/data/CASS_50m_256x256x139_600CCN/lwcs_processed"
# files = glob.glob(join(data_dir, "*.npz"))
# N_cloud = 110
i=0
# for file in files:
beta_cloud = np.reshape(np.array((np.arange(1,33))**0.5)*10,(1,1,32))
beta_cloud = beta_cloud.astype(float_reg)
beta_max = beta_cloud.max()
print(beta_cloud.mean())

# Grid parameters #
# bounding box
voxel_size_x = 0.5
voxel_size_y = 0.5
voxel_size_z = 0.04
edge_x = 1
edge_y = 1
edge_z = voxel_size_z * 32
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
height_factor = 2

focal_length = 10e-3
sensor_size = np.array((50e-3, 50e-3)) / height_factor
ps_max = 76

pixels = np.array((ps_max, ps_max))

N_cams = 1
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
volume_center = bbox[:, 1] * np.unravel_index(np.argmax(beta_cloud, axis=None), beta_cloud.shape) / np.array(beta_cloud.shape)
R = height_factor * edge_z

for cam_ind in range(N_cams - 1):
    cam_deg = 360 // (N_cams - 1)
    theta = 29
    theta_rad = theta * (np.pi / 180)
    phi = (-(N_cams // 2) + cam_ind) * cam_deg
    phi_rad = phi * (np.pi / 180)
    t = R * theta_phi_to_direction(theta_rad, phi_rad) + volume_center
    euler_angles = np.array((180 - theta, 0, phi - 90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
t = R * theta_phi_to_direction(0, 0) + volume_center
euler_angles = np.array((180, 0, -90))
cameras.append(Camera(t, euler_angles, focal_length, sensor_size, pixels))
# Simulation parameters
N_render_gt = 10
Np_gt = int(1e5)
# Np_gt *= N_render_gt

Np_max = int(1e3)
Np = Np_max

resample_freq = 1
step_size = 7e4
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.05
iterations = 50
# to_mask = True
tensorboard = False
tensorboard_freq = 10
win_size = 100

beta_cloud = beta_cloud.astype(float_reg)
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
scene_rr = SceneRR_batch(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob, N_batches=1)

for i in range(N_render_gt):
    cuda_paths = scene_rr.build_paths_list(Np_gt)
    if i == 0:
        I_gt = scene_rr.render(cuda_paths)
    else:
        I_gt += scene_rr.render(cuda_paths)
I_gt /= N_render_gt
scene_rr = SceneRR_batch(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob, N_batches=50)
del (cuda_paths)
cuda_paths = None
max_val = np.max(I_gt, axis=(1,2))
print("Calculating Cloud Mask")
# cloud_mask = scene_rr.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
cloud_mask = beta_cloud > 0.01
# mask_grader(cloud_mask, beta_gt>0.01, beta_gt)
scene_rr.set_cloud_mask(cloud_mask)
alpha = 0.9
beta1 = 0.9
beta2 = 0.999
scaling_factor = 1.5
# optimizer = MomentumSGD(volume, step_size, alpha, beta_max, beta_max)
optimizer = SGD(volume, step_size, beta_max, beta_max)
scene_rr.upscale_cameras(ps_max)

ps = ps_max
r = 1 / np.sqrt(scaling_factor)
n = int(np.ceil(np.log(ps / ps_max) / np.log(r)))
non_min_couter = 0
next_phase = False
min_loss = 1  #
upscaling_counter = 0

# Initialization
beta_init = np.zeros_like(beta_cloud)
beta_init[volume.cloud_mask] = 2

##################################
            # SGD
##################################
volume.set_beta_cloud(beta_init)
beta_opt = volume.beta_cloud
loss = 1
start_loop = time()
grad_gd = []
cov_gd = []
correlation_gd = []
grad_norm_gd = []
eps_gd = []
for iter in range(iterations):
    # cloud['est_cloud{}'.format(iter)] = beta_opt.tolist()

    # if iter > start_iter:
    #     resample_freq = 1
    print(f"\niter {iter}")
    beta_opt = volume.beta_cloud
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    max_dist = np.max(abs_dist)
    rel_dist1 = relative_distance(beta_cloud, beta_opt)
    eps_gd.append(rel_dist1)
    print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")

    scene_rr.init_cuda_param(Np)
    print("RESAMPLING PATHS ")
    start = time()
    del (cuda_paths)
    cuda_paths = scene_rr.build_paths_list(int(Np))
    end = time()
    print(f"building path list took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=I_gt)
    total_grad *= (ps * ps)
    end = time()
    print(f"rendering took: {end - start}")

    norm_grad = np.linalg.norm(total_grad.ravel(), ord=2)
    grad_gd.append(np.mean(np.squeeze(total_grad),0))
    cov_gd.append(np.cov(np.squeeze(total_grad).T).T)
    correlation_gd.append(np.corrcoef(np.squeeze(total_grad).T, dtype=total_grad.dtype))
    if iter % 10==0:
        plt.imshow(correlation_gd[-1])
        plt.colorbar()
        plt.clim(-1, 1)
        plt.show()
    total_grad = np.mean(total_grad,0)
    # damped_correlation = np.eye(correlation_gd[-1].shape[0], dtype=total_grad.dtype) + 0.2 * (correlation_gd[-1] - np.eye(correlation_gd[-1].shape[0], dtype=total_grad.dtype))
    optimizer.step(total_grad)
grad_gd = np.array(grad_gd).reshape(iterations,-1)


##################################
            # Correlation
##################################
volume.set_beta_cloud(beta_init)
beta_opt = volume.beta_cloud
loss = 1
start_loop = time()
grad = []
cov = []
correlation = []
grad_norm = []
eps = []
for iter in range(iterations):
    # cloud['est_cloud{}'.format(iter)] = beta_opt.tolist()

    # if iter > start_iter:
    #     resample_freq = 1
    print(f"\niter {iter}")
    beta_opt = volume.beta_cloud
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    max_dist = np.max(abs_dist)
    rel_dist1 = relative_distance(beta_cloud, beta_opt)
    eps.append(rel_dist1)
    print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")

    scene_rr.init_cuda_param(Np)
    print("RESAMPLING PATHS ")
    start = time()
    del (cuda_paths)
    cuda_paths = scene_rr.build_paths_list(int(Np))
    end = time()
    print(f"building path list took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=I_gt)
    total_grad *= (ps * ps)
    end = time()
    print(f"rendering took: {end - start}")

    norm_grad = np.linalg.norm(total_grad.ravel(), ord=2)
    grad.append(np.mean(np.squeeze(total_grad),0))
    cov.append(np.cov(np.squeeze(total_grad).T).T)
    correlation.append(np.corrcoef(np.squeeze(total_grad).T, dtype=total_grad.dtype))
    if iter % 10==0:
        plt.imshow(correlation[-1])
        plt.colorbar()
        plt.clim(-1, 1)
        plt.show()
    total_grad = np.mean(total_grad,0)
    damped_correlation = np.eye(correlation[-1].shape[0], dtype=total_grad.dtype) + 0.2 * (correlation[-1] - np.eye(correlation[-1].shape[0], dtype=total_grad.dtype))
    total_grad = (damped_correlation @ total_grad.ravel()).reshape((1,1,-1)) / 2
    optimizer.step(total_grad)
grad = np.array(grad).reshape(iterations,-1)


plt.plot(eps, label='Correlation')
plt.plot(eps_gd, label='GD')
plt.legend()
plt.show()
# corr_norm[corr_norm != 0] = corr_norm[corr_norm != 0] / (np.linalg.norm(grad_norm, ord=2, axis=0) * np.linalg.norm(max_grad_norm, ord=2))[corr_norm != 0]

scipy.io.savemat(os.path.join('/home/roironen/Path_Recycling/code/projects/correlation/output', 'correlation{}.mat'.format(i)),
      {'beta': beta_cloud,
       'shape': beta_cloud.shape,
       'cov': cov,
        'correlation': correlation,
       })

print()