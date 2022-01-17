import shutil
import os, sys
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
if False:
    data_dir = "/home/roironen/Path_Recycling/code/generated_data_noise2clean"
    files = [f for f in glob.glob(os.path.join(data_dir, 'cloud*'))]

    for file in files:
        with open(glob.glob(os.path.join(file, 'cloud_info.pkl'))[0], 'rb') as f:
            x = pickle.load(f)
        shape = x['shape']
        beta = x['beta_gt']
        grad = []
        for iter in range(100):
            path = glob.glob(os.path.join(file, 'total_grad_it_{}_Np_10000000_*.pkl'.format(iter)))[0]
            with open(path, 'rb') as f:
                grad.append(np.squeeze(pickle.load(f).toarray()))

        grad = np.array(grad)
        max_grad = grad[:, np.argmax(beta)]  # np.ravel_multi_index(,shape)
        corr = grad.T @ max_grad
        corr[corr != 0] = corr[corr != 0] / (np.linalg.norm(grad, ord=2, axis=0) * np.linalg.norm(max_grad, ord=2))[
            corr != 0]
        scipy.io.savemat(os.path.join(file, 'correlation.mat'),
                         {'beta': beta.toarray(),
                          'shape': shape,
                          'corr_ind': np.argmax(beta),
                          'corr': corr})
        print()

else:
    sun_angles = np.array([180, 0]) * (np.pi / 180)
    data_dir = "/home/roironen/pytorch3d/projects/CT/data/CASS_50m_256x256x139_600CCN/lwcs_processed"
    files = glob.glob(join(data_dir, "*.npz"))
    N_cloud = 110
    i=0
    for file in files:
        beta_cloud = np.load(file)['lwc'] * 100
        beta_cloud = beta_cloud.astype(float_reg)
        beta_max = beta_cloud.max()
        print(beta_cloud.mean())
        if beta_cloud.mean() < 0.1:
            continue
        i +=1
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
        Np_gt = int(1e7)
        # Np_gt *= N_render_gt

        Np_max = int(1e6)
        Np = Np_max

        resample_freq = 1
        step_size = 7e8
        # Ns = 15
        rr_depth = 20
        rr_stop_prob = 0.05
        iterations = 800
        # to_mask = True
        tensorboard = False
        tensorboard_freq = 10
        win_size = 100

        beta_cloud = beta_cloud.astype(float_reg)
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
        optimizer = MomentumSGD(volume, step_size, alpha, beta_max, beta_max)
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

        volume.set_beta_cloud(beta_init)
        beta_opt = volume.beta_cloud
        loss = 1
        start_loop = time()
        grad = []
        grad_norm = []
        for iter in range(iterations):
            # cloud['est_cloud{}'.format(iter)] = beta_opt.tolist()

            # if iter > start_iter:
            #     resample_freq = 1
            print(f"\niter {iter}")
            abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
            max_dist = np.max(abs_dist)
            rel_dist1 = relative_distance(beta_cloud, beta_opt)

            print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")

            for i, Np_factor in enumerate([1]):
                scene_rr.init_cuda_param(Np)
                print("RESAMPLING PATHS ")
                start = time()
                del (cuda_paths)
                cuda_paths = scene_rr.build_paths_list(int(Np * Np_factor))
                end = time()
                print(f"building path list took: {end - start}")
                # differentiable forward model
                start = time()
                I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=I_gt)
                total_grad *= (ps * ps)
                end = time()
                print(f"rendering took: {end - start}")

                # dif = (I_opt - I_gt).reshape(1,1,1, N_cams, *scene_rr.pixels_shape)
                # grad_norm = np.linalg.norm(total_grad)
                # if i==1:
                #     name = "noise"
                # else:
                #     name = "clean"
                # cloud['total_grad_it_{}_{}'.format(iter, name)] = csr_matrix(total_grad.ravel())#.tolist()
            norm_grad = np.linalg.norm(total_grad.ravel(), ord=2)
            grad.append(np.squeeze(total_grad))
            if norm_grad == 0:
                grad_norm.append(np.squeeze(total_grad))
            else:
                grad_norm.append(np.squeeze(total_grad)/norm_grad)
            optimizer.step(total_grad)
        grad = np.array(grad).reshape(iterations,-1)
        grad_norm = np.array(grad_norm).reshape(iterations,-1)
        max_grad = grad[:,np.argmax(beta_cloud.ravel())] #np.ravel_multi_index(,shape)
        max_grad_norm = grad_norm[:,np.argmax(beta_cloud.ravel())] #np.ravel_multi_index(,shape)

        corr = grad.T @ max_grad
        corr_norm = grad_norm.T @ max_grad_norm
        corr[corr != 0] = corr[corr != 0] / (np.linalg.norm(grad, ord=2, axis=0) * np.linalg.norm(max_grad, ord=2))[corr != 0]
        # corr_norm[corr_norm != 0] = corr_norm[corr_norm != 0] / (np.linalg.norm(grad_norm, ord=2, axis=0) * np.linalg.norm(max_grad_norm, ord=2))[corr_norm != 0]

        scipy.io.savemat(os.path.join('/home/roironen/Path_Recycling/code/projects/correlation/output', 'correlation{}.mat'.format(i)),
              {'beta': beta_cloud,
               'shape': beta_cloud.shape,
               'corr_ind': np.argmax(beta_cloud),
               'corr': corr.reshape(beta_cloud.shape),
               })

        print()