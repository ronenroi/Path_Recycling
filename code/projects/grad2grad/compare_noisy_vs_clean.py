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


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
# data_dir = "/home/roironen/pytorch3d/projects/CT/data/CASS_50m_256x256x139_600CCN/lwcs_processed"
# files = glob.glob(join(data_dir,"*.npz"))
N_cloud = 100

# beta_cloud = np.load(files[0])['lwc']
# beta_cloud = beta_cloud.astype(float_reg)
Nz = 2
# Grid parameters #
# bounding box
voxel_size_x = 0.01
voxel_size_y = 0.01
voxel_size_z = 0.01
edge_x = voxel_size_x * 1
edge_y = voxel_size_y * 1
edge_z = voxel_size_z * Nz
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])


print([1,1,Nz])
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
R = height_factor * edge_z


cam_deg = 360 // (N_cams)
for cam_ind in range(N_cams):
    theta = 60
    theta_rad = theta * (np.pi/180)
    phi = (-(N_cams//2) + cam_ind) * cam_deg
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta_rad,phi_rad) + volume_center
    euler_angles = np.array((180-theta, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)
# t = R * theta_phi_to_direction(90,0) + volume_center
# euler_angles = np.array((180, 0, -90))
# cameras.append(Camera(t, euler_angles, focal_length, sensor_size, pixels))

# Simulation parameters
N_render_gt = 5
Np_gt = int(1e7)
# Np_gt *= N_render_gt

Np_max = int(1e4)
Np = Np_max


resample_freq = 1
step_size = 5e0
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.05
iterations = 30
# to_mask = True
tensorboard = False
tensorboard_freq = 10
win_size = 100

i_cloud = 0
for i_cloud in range(1):

    beta_cloud = np.random.rand(1,1,Nz)*100
    beta_max = 100

    print(beta_cloud.mean())
    beta_cloud = beta_cloud.astype(float_reg)
    # Declerations
    grid = Grid(bbox, beta_cloud.shape)
    volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
    beta_gt = np.copy(beta_cloud)
    scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)



    for i in range(N_render_gt):
        cuda_paths = scene_rr.build_paths_list(Np_gt)
        if i == 0:
            I_gt = scene_rr.render(cuda_paths)
        else:
            I_gt += scene_rr.render(cuda_paths)
    I_gt /= N_render_gt

    del(cuda_paths)
    cuda_paths = None
    max_val = np.max(I_gt, axis=(1,2))
    print("Calculating Cloud Mask")
    # cloud_mask = scene_rr.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
    cloud_mask =beta_cloud > 0.00
    # mask_grader(cloud_mask, beta_gt>0.01, beta_gt)
    scene_rr.set_cloud_mask(cloud_mask)
    # beta_scalar_init = scene_rr.find_best_initialization(beta_gt, I_gt,0,30,10,Np_gt,True)

    scene_rr.init_cuda_param(Np)
    alpha = 0.9
    beta1 = 0.9
    beta2 = 0.999
    # start_iter = -1
    scaling_factor = 1.5
    ps = ps_max
    r = 1/np.sqrt(scaling_factor)
    n = int(np.ceil(np.log(ps/ps_max)/np.log(r)))

    I_gts = [I_gt]
    pss = [ps_max]
    # print("creating I_gt pyramid...")
    # for iter in tqdm(range(n)):
    #
    #     if iter < n-1:
    #         ps_temp = int(ps_max * r**iter)
    #         temp = ps_temp/ps_max
    #     else:
    #         temp = ps/ps_max
    #         ps_temp = ps
    #     I_temp = zoom(I_gt, (1,temp, temp), order=1)
    #     I_temp *= 1/(temp*temp) # light correction factor
    #     I_gts.insert(0,I_temp)
    #     pss.insert(0, ps_temp)

    I_gt = I_gts[0]
    ps = pss[0]
    print(pss)
    scene_rr.upscale_cameras(ps)


    # grad_norm = None
    non_min_couter = 0
    min_loss = 1#
    upscaling_counter = 0
    noisy_err_list = []
    clean_err_list = []
    noisy_time_list = []
    clean_time_list = []

    # Initialization
    beta_init = np.zeros_like(beta_cloud)
    beta_init[volume.cloud_mask] = 2

    volume.set_beta_cloud(beta_init)
    beta_opt = volume.beta_cloud
    # optimizer = SGD(volume,step_size)
    optimizer = MomentumSGD(volume, step_size, alpha, beta_max, beta_max)
    # optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_max, beta_max, 1)

    loss = 1
    start_loop = time()
    grad = None
    err = []
    lost_list = []
    time_list = []
    grad_dic ={}
    for iter in range(iterations):
        # cloud['est_cloud{}'.format(iter)] = beta_opt.tolist()

        # if iter > start_iter:
        #     resample_freq = 1
        print(f"\niter {iter}")
        abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
        max_dist = np.max(abs_dist)
        rel_dist1 = relative_distance(beta_cloud, beta_opt)

        print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")

        scales = [1]*200 + [10]*200
        lr_grads = []
        hr_grads = []
        grad_sum = np.zeros_like(beta_cloud)
        first =True
        grad = 0
        for state in scales:
            print("RESAMPLING PATHS ")
            start = time()
            del(cuda_paths)
            cuda_paths = scene_rr.build_paths_list(int(Np*state))
            end = time()
            print(f"building path list took: {end - start}")
            # differentiable forward model
            start1 = time()
            I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=I_gt)
            total_grad *= (ps*ps)
            end = time()
            print(f"rendering took: {end-start1}")

            dif = (I_opt - I_gt).reshape(1,1,1, N_cams, *scene_rr.pixels_shape)
            grad_norm = np.linalg.norm(total_grad)
            if state == 1:
                lr_grads.append(total_grad.copy())
            else:
                grad +=total_grad.copy()
                hr_grads.append(total_grad.copy())
            # label = 'noisy' if state == 1 and first else None if state == 1 else 'clean' if state==20 else 'super clean'
            # plt.plot([0,np.squeeze(total_grad)[0]],[0,np.squeeze(total_grad)[1]],'b' if state==1 else 'r' if state==20 else 'k',label = label)
            first = False
        grad /= 200
        # grad_sum /= 20
        # plt.plot([0, np.squeeze(grad_sum)[0]], [0, np.squeeze(grad_sum)[1]], 'g',label='noisy average')
        lr_mean = np.squeeze(np.mean(np.array(lr_grads),0))
        norm = np.linalg.norm(lr_mean)
        lr_mean /= norm
        lr_std = np.squeeze(np.std(np.array(lr_grads),0)) / norm
        lr1 = lr_mean.copy()
        lr1[0] += np.sign(lr_mean[0]) * lr_std[0]
        lr1[1] -=  np.sign(lr_mean[1]) * lr_std[1]
        lr2 = lr_mean.copy()
        lr2[0] -=  np.sign(lr_mean[0]) * lr_std[0]
        lr2[1] +=  np.sign(lr_mean[1]) * lr_std[1]

        norm = np.linalg.norm(np.squeeze(np.mean(np.array(hr_grads),0)))
        hr_mean = np.squeeze(np.mean(np.array(hr_grads),0)) / norm
        hr_std = np.squeeze(np.std(np.array(hr_grads), 0)) / norm
        hr1 = hr_mean.copy()
        hr1[0] +=  np.sign(hr_mean[0]) * hr_std[0]
        hr1[1] -= np.sign(hr_mean[1]) * hr_std[1]
        hr2 = hr_mean.copy()
        hr2[0] -= np.sign(hr_mean[0]) * hr_std[0]
        hr2[1] += np.sign(hr_mean[1]) * hr_std[1]
        fig, ax = plt.subplots(1)
        ax.plot([0, lr_mean[0]], [0, lr_mean[1]], color='blue',label='Noisy gradient average')
        ax.plot([0, hr_mean[0]], [0, hr_mean[1]], color='red', label='Clean gradient average')
        # ax.fill_between([0, lr_mean[0]], [0, lr_mean[0]+lr_std[0]], [0, lr_mean[1]+lr_std[1]], facecolor='blue', alpha=0.5)
        # ax.fill_between([0, hr_mean[0]], [0, hr_mean[0]+hr_std[0]], [0, hr_mean[1]+hr_std[1]], facecolor='yellow', alpha=0.5)
        ax.fill([0,lr1[0], lr2[0]], [0, lr1[1], lr2[1]], facecolor='lightblue', edgecolor='blue', linewidth=3, alpha=0.5)
        ax.fill([0,hr1[0], hr2[0]], [0, hr1[1], hr2[1]], facecolor='lightsalmon', edgecolor='orangered', linewidth=3, alpha=0.5)

        # ax.fill_between(t, mu2 + sigma2, mu2 - sigma2, facecolor='yellow', alpha=0.5)
        # ax.set_title(r'random walkers empirical $\mu$ and $\pm \sigma$ interval')
        # ax.legend(loc='upper left')
        # ax.set_xlabel('num steps')
        # ax.set_ylabel('position')
        ax.grid()
        plt.legend()
        # plt.legend(['noisy']*20 + ['clean', 'noisy average'])
        plt.xlabel("g1")
        plt.ylabel("g2")
        plt.title(f"Iteration = {iter}")
        plt.show()
        optimizer.step(grad)
        time_list.append(time()-start)
        # print("gradient step took:",time()-start)
        loss = 0.5 * np.sum(dif * dif)
        lost_list.append(loss)
        # loss = 0.5 * np.sum(np.abs(dif))
        # if loss < min_loss:
        #     min_loss = loss
        #     non_min_couter = 0
        # else:
        #     non_min_couter += 1
        # print(f"loss = {loss}, grad_norm={grad_norm}, max_grad={np.max(total_grad)}")
        err.append(rel_dist1)
    plt.plot(err)
    plt.xlabel("Iterations")
    plt.ylabel("Relative error")
    plt.show()

    plt.plot(lost_list)
    plt.xlabel("Iterations")
    plt.ylabel("Image loss")
    plt.show()


