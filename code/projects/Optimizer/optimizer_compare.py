import shutil
import os, sys

import numpy as np

my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene_rr_noNEgrad import *
# from classes.scene_rr import *
# from classes.scene_seed import SceneRR_noNE
# from classes.scene_seed_roi import *

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


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
data_dir = "/home/roironen/pytorch3d/projects/CT/data/CASS_50m_256x256x139_600CCN/lwcs_processed"
files = glob.glob(join(data_dir,"*.npz"))
# N_cloud = 110

beta_cloud = np.load(files[0])['lwc']
beta_cloud = beta_cloud.astype(float_reg)

# Grid parameters #
# bounding box
voxel_size_x = 0.05
voxel_size_y = 0.05
voxel_size_z = 0.04
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
height_factor = 2

focal_length = 1e-4 #####################
sensor_size = np.array((3e-4, 3e-4)) / height_factor #####################
ps_max = 76

pixels = np.array((ps_max, ps_max))

N_cams = 9
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
volume_center[-1] *= 1.8
R = edge_z * 2

cam_deg = 360 // (N_cams-1)
for cam_ind in range(N_cams-1):
    theta = 90
    theta_rad = theta * (np.pi/180)
    phi = (-(N_cams//2) + cam_ind) * cam_deg
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi_rad) + volume_center
    euler_angles = np.array((180-theta, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
t = R * theta_phi_to_direction(0,0) + volume_center
euler_angles = np.array((180, 0, -90))
cameras.append(Camera(t, euler_angles, cameras[0].focal_length, cameras[0].sensor_size, cameras[0].pixels))


# mask parameters
image_threshold = 0.15
hit_threshold = 0.9
spp = 100000

# Simulation parameters
N_render_gt = 10
Np_gt = int(1e7)
# Np_gt *= N_render_gt

Np_max = int(1e5)
Np = Np_max


resample_freq = 1
# step_size = 1
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.05
iterations = 300
# to_mask = True
tensorboard = False
tensorboard_freq = 10
win_size = 100

i_cloud = 0
for file in files:
    beta_cloud = np.load(file)['lwc'] * 100

    print(beta_cloud.mean())
    if np.sum(beta_cloud>1)< 500:
        continue
    # beta_cloud[beta_cloud==0]=np.nan
    # beta_cloud = np.nanmean(beta_cloud, axis=(0, 1))[None,None]
    # beta_cloud[np.isnan(beta_cloud)] = 0
    beta_max = beta_cloud.max()
    cloud_mask = beta_cloud>0

    beta_cloud = beta_cloud.astype(float_reg)
    # Declerations
    grid = Grid(bbox, beta_cloud.shape)
    volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
    volume.set_mask(cloud_mask)
    beta_gt = np.copy(beta_cloud)
    # scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
    # scene_rr = SceneRR_noNE(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
    scene_seed = SceneRR_noNE_batched(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
    visual = Visual_wrapper(scene_seed)
    # visual.create_grid()
    # visual.plot_cameras()
    # visual.plot_medium()
    plt.show()
    scene_seed.init_cuda_param(Np_gt, init=True)
    for i in range(N_render_gt):
        cuda_paths = scene_seed.build_paths_list(Np_gt)
        if i == 0:
            I_gt = scene_seed.render(cuda_paths)
        else:
            I_gt += scene_seed.render(cuda_paths)
    I_gt /= N_render_gt


    max_val = np.max(I_gt, axis=(1,2))
    visual.plot_images(I_gt, "GT")
    plt.show()
    # cloud['I_gt'] =  csr_matrix(I_gt.ravel())#.tolist()

    print("Calculating Cloud Mask")
    # cloud_mask = scene_rr.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
    # mask_grader(cloud_mask, beta_gt>0.01, beta_gt)

    # beta_scalar_init = scene_rr.find_best_initialization(beta_gt, I_gt,0,30,10,Np_gt,True)
    Momentum_err = []
    AdaptiveStd_err = []
    SGD_err = []
    AdaptiveStd_loss = []
    Momentum_loss = []
    SGD_loss = []
    for state in ['Momentum', 'SGD', 'STD']:#
        alpha = 0.9
        beta1 = 0.9
        beta2 = 0.999
        ps = ps_max

        I_gts = [I_gt]
        pss = [ps_max]

        I_gt = I_gts[0]
        ps = pss[0]
        print(pss)



        # grad_norm = None
        non_min_couter = 0
        next_phase = False
        min_loss = 1#
        upscaling_counter = 0

        # Initialization
        beta_init = np.zeros_like(beta_cloud)
        beta_init[volume.cloud_mask] = 2

        volume.set_beta_cloud(beta_init)
        loss = 1
        start_loop = time()
        grad = None
        err = []
        loss_list = []
        if state == 'Momentum':
            step_size = 1e9
            optimizer = MomentumSGD(volume, step_size, alpha, beta_max, beta_max)
            flag = False
            N_batch = 1

        elif state == 'SGD':
            step_size = 1e9
            optimizer = SGD(volume, step_size, beta_max, beta_max)
            flag = False
            N_batch = 1
        else:
            step_size = 1e9
            optimizer = AdaptiveStdSGD(volume, step_size, beta1, beta2, beta_max, beta_max)
            flag = True
            N_batch = 100
        scene_seed = SceneRR_noNE_batched(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob,N_batches=N_batch)
        scene_seed.init_cuda_param(Np, init=True)
        scene_seed.set_cloud_mask(cloud_mask)

        for iter in range(iterations):
            # cloud['est_cloud{}'.format(iter)] = beta_opt.tolist()
            beta_opt = volume.beta_cloud

            # if iter > start_iter:
            #     resample_freq = 1
            print(f"\niter {iter}")
            abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
            max_dist = np.max(abs_dist)
            rel_dist1 = relative_distance(beta_cloud, beta_opt)
            err.append(rel_dist1)


            print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")


            print("RESAMPLING PATHS ")
            start = time()
            cuda_paths = scene_seed.build_paths_list(int(Np), to_sort=False)
            end = time()
            print(f"building path list took: {end - start}")
            # differentiable forward model
            start = time()
            I_opt, total_grad = scene_seed.render(cuda_paths, I_gt=I_gt)
            total_grad *= (ps * ps)
            end = time()
            dif = (I_opt - I_gt).reshape(1, 1, 1, N_cams, *scene_seed.pixels_shape)
            loss = 0.5 * np.sum(dif * dif)
            loss_list.append(loss)
            if flag:
                grad_lr = total_grad
                total_grad = np.mean(total_grad,0)
                grad_lr[grad_lr==0] = np.nan
                optimizer.std = np.nanstd(grad_lr,0)
                optimizer.std[np.isnan(optimizer.std)] = 0
            # print(optimizer.std[cloud_mask])
            else:
                total_grad = np.squeeze(total_grad)
            optimizer.step(total_grad)
        if state == 'Momentum':
            Momentum_err.append(err)
            Momentum_loss.append(loss_list)
        elif state == 'SGD':
            SGD_err.append(err)
            SGD_loss.append(loss_list)
        # optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_max, beta_max, 1)
        else:
            AdaptiveStd_err.append(err)
            AdaptiveStd_loss.append(loss_list)
            std_list = optimizer.scalegrad

    i_cloud += 1
    plt.plot(Momentum_err[-1])
    plt.plot(SGD_err[-1])
    plt.plot(AdaptiveStd_err[-1])
    plt.legend(['Mumentum', 'SGD', 'AdaptiveStd'])
    plt.xlabel("Iterations")
    plt.ylabel("Epsilon - Relative error")
    plt.show()

    plt.plot(Momentum_loss[-1])
    plt.plot(SGD_loss[-1])
    plt.plot(AdaptiveStd_loss[-1])
    plt.legend(['Mumentum', 'SGD', 'AdaptiveStd'])
    plt.xlabel("Iterations")
    plt.ylabel("L2 loss")
    plt.yscale('log')
    plt.show()

    plt.plot(std_list)
    plt.legend(['scales'])
    plt.xlabel("Iterations")
    plt.ylabel("Gradient scales")

    plt.show()
    #################################
    # altitude = np.linspace(0, voxel_size_z * beta_cloud.shape[2], beta_cloud.shape[2])
    # b = np.squeeze(np.copy(beta_opt))
    # b[b==0] = np.nan
    # b = np.nanmean(b,axis=(0,1))
    #
    # b_gt = np.squeeze(np.copy(beta_cloud))
    # b_gt[b_gt==0] = np.nan
    # b_gt = np.nanmean(b_gt,axis=(0,1))
    # plt.plot(b, altitude)
    # plt.plot(b_gt, altitude)
    # plt.legend(['Estimated', 'GT'])
    # plt.xlabel("Extinction [1/km]")
    # plt.ylabel("Altitude")
    # plt.show()
