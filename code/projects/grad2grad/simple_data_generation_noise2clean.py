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
N_cloud = 110

beta_cloud = np.load(files[0])['lwc']
beta_cloud = beta_cloud.astype(float_reg)

# Grid parameters #
# bounding box
voxel_size_x = 0.5
voxel_size_y = 0.5
voxel_size_z = 0.04
edge_x = voxel_size_x * 1
edge_y = voxel_size_y * 1
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

N_cams = 3
cameras = []
volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
R = height_factor * edge_z

cam_deg = 360 // (N_cams-1)
for cam_ind in range(N_cams-1):
    theta = 29
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
step_size = 7e4
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.05
iterations = 50
# to_mask = True
tensorboard = False
tensorboard_freq = 10
win_size = 100

i_cloud = 0
is_continue = False
for file in files:
    beta_cloud = np.load(file)['lwc'] * 100
    beta_max = beta_cloud.max()


    print(beta_cloud.mean())
    if beta_cloud.mean() < 0.1:
        continue
    else:
        break

cloud_mask = np.sum(beta_cloud,axis=(0,1))>0
for i_cloud in range(110):
    output_dir = "simple_generated_data_noise2clean/cloud{}".format(i_cloud)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    cloud = {'file': file}
    beta_cloud = cloud_mask * np.random.rand(1) * np.linspace(1,32,32)
    beta_cloud = np.reshape(beta_cloud,(1,1,32))
    beta_cloud = beta_cloud.astype(float_reg)
    # Declerations
    grid = Grid(bbox, (1,1,32))
    volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
    beta_gt = np.copy(beta_cloud)
    scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
    cloud['beta_gt'] = beta_gt#.tolist()
    cloud['shape'] = beta_gt.shape

    with open('{}/cloud_info.pkl'.format(output_dir), 'wb') as f:
        print(f"saving cloud info")
        pickle.dump(cloud, f, protocol=pickle.HIGHEST_PROTOCOL)
    visual = Visual_wrapper(scene_rr)
    # visual.create_grid()
    # visual.plot_cameras()
    # visual.plot_medium()
    plt.show()

    for i in range(N_render_gt):
        cuda_paths = scene_rr.build_paths_list(Np_gt)
        if i == 0:
            I_gt = scene_rr.render(cuda_paths)
        else:
            I_gt += scene_rr.render(cuda_paths)
    I_gt /= N_render_gt

    del(cuda_paths)
    with open('{}/I_gt.pkl'.format(output_dir), 'wb') as f:
        print(f"saving cloud images")
        pickle.dump(csr_matrix(I_gt.ravel()), f, protocol=pickle.HIGHEST_PROTOCOL)
    cuda_paths = None
    max_val = np.max(I_gt, axis=(1,2))
    visual.plot_images(I_gt, "GT")
    plt.show()
    # cloud['I_gt'] =  csr_matrix(I_gt.ravel())#.tolist()

    print("Calculating Cloud Mask")
    # cloud_mask = scene_rr.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
    cloud_mask = beta_cloud > 0.01
    # mask_grader(cloud_mask, beta_gt>0.01, beta_gt)
    scene_rr.set_cloud_mask(cloud_mask)
    # beta_scalar_init = scene_rr.find_best_initialization(beta_gt, I_gt,0,30,10,Np_gt,True)
    # cloud['cloud_mask'] = csr_matrix(cloud_mask.ravel())#.tolist()
    if not is_continue:
        with open('{}/cloud_mask.pkl'.format(output_dir), 'wb') as f:
            print(f"saving cloud mask")
            pickle.dump(cloud_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
    alpha = 0.9
    beta1 = 0.9
    beta2 = 0.999
    # start_iter = -1
    scaling_factor = 1.5
    # optimizer = SGD(volume,step_size)
    optimizer = MomentumSGD(volume, step_size, alpha, beta_max, beta_max)
    # optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_max, beta_max, 1)

    ps = ps_max
    r = 1/np.sqrt(scaling_factor)
    n = int(np.ceil(np.log(ps/ps_max)/np.log(r)))

    I_gts = [I_gt]
    pss = [ps_max]

    I_gt = I_gts[0]
    ps = pss[0]
    print(pss)
    scene_rr.upscale_cameras(ps)


    # grad_norm = None
    non_min_couter = 0
    next_phase = False
    min_loss = 1#
    upscaling_counter = 0

    # Initialization
    beta_init = np.zeros_like(beta_cloud)
    beta_init[volume.cloud_mask] = 2

    volume.set_beta_cloud(beta_init)
    beta_opt = volume.beta_cloud
    loss = 1
    start_loop = time()
    grad = None
    for iter in range(iterations):
        # cloud['est_cloud{}'.format(iter)] = beta_opt.tolist()

        # if iter > start_iter:
        #     resample_freq = 1
        print(f"\niter {iter}")
        abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
        max_dist = np.max(abs_dist)
        rel_dist1 = relative_distance(beta_cloud, beta_opt)

        print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")
        grad_lr = np.zeros((10,*beta_cloud.shape))
        for i, Np_factor in enumerate([0.1]*10 + [10]):
            scene_rr.init_cuda_param(Np)
            print("RESAMPLING PATHS ")
            start = time()
            del(cuda_paths)
            cuda_paths = scene_rr.build_paths_list(int(Np*Np_factor))
            end = time()
            print(f"building path list took: {end - start}")
            # differentiable forward model
            start = time()
            I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=I_gt)
            total_grad *= (ps*ps)
            end = time()
            print(f"rendering took: {end-start}")
            if Np_factor == 0.1:
                grad_lr[i] = total_grad
        with open('{}/total_grad_it_{}_Np_{}.pkl'.format(output_dir,iter,int(Np)), 'wb') as f:
            print(f"saving cloud grad")
            pickle.dump(grad_lr, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open('{}/total_grad_it_{}_Np_{}.pkl'.format(output_dir,iter,int(Np*10)), 'wb') as f:
            print(f"saving cloud grad")
            pickle.dump(total_grad, f, protocol=pickle.HIGHEST_PROTOCOL)
        start = time()

        optimizer.step(total_grad)

