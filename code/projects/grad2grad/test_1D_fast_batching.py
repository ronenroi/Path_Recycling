import numpy as np

from projects.grad2grad.noise2noise import Noise2Noise
from projects.grad2grad.datasets import load_dataset
from argparse import ArgumentParser

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
# import matplotlib
# matplotlib.use('Agg')

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

import torch

def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='Test 1D cloud dataset')

    # Data parameters
    parser.add_argument('-v', '--valid-dir', help='test set path', default='/home/roironen/Path_Recycling/code/data/1D_generated_data_noise2clean/val')
    parser.add_argument('--cuda', help='use cuda', action='store_false')
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('--image-size', help='image size', default=76, type=int)
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['mc'], default='mc', type=str)
    parser.add_argument('-net', help='network type', default='Linear', type=str)


    return parser.parse_args()


if __name__ == '__main__':
    """Test Noise2Noise."""

    # Parse training parameters
    parse_params = parse_args()
    files = [f for f in glob.glob(os.path.join(parse_params.valid_dir, '*'))]

    ########################
    # Atmosphere parameters#
    ########################
    sun_angles = np.array([180, 0]) * (np.pi / 180)

    #####################
    # Volume parameters #
    #####################
    # construct betas
    with open(glob.glob(os.path.join(files[0], 'cloud_info.pkl'))[0], 'rb') as f:
        x = pickle.load(f)
    shape = x['shape']
    beta_cloud = x['beta_gt'].reshape(shape)
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

    focal_length = 1e-4  #####################
    sensor_size = np.array((3e-4, 3e-4)) / height_factor  #####################
    ps_max = 76

    pixels = np.array((ps_max, ps_max))

    N_cams = 9
    cameras = []
    volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
    volume_center[-1] *= 1.8
    R = edge_z * 2

    cam_deg = 360 // (N_cams - 1)
    for cam_ind in range(N_cams - 1):
        theta = 90
        theta_rad = theta * (np.pi / 180)
        phi = (-(N_cams // 2) + cam_ind) * cam_deg
        phi_rad = phi * (np.pi / 180)
        t = R * theta_phi_to_direction(theta_rad, phi_rad) + volume_center
        euler_angles = np.array((180 - theta, 0, phi - 90))
        camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
        cameras.append(camera)
    t = R * theta_phi_to_direction(0, 0) + volume_center
    euler_angles = np.array((180, 0, -90))
    cameras.append(Camera(t, euler_angles, cameras[0].focal_length, cameras[0].sensor_size, cameras[0].pixels))

    # Simulation parameters
    N_render_gt = 10
    Np_gt = int(1e7)
    # Np_gt *= N_render_gt

    Np_max = int(1e3)
    Np = Np_max

    resample_freq = 1
    step_size = 5e4
    # Ns = 15
    rr_depth = 20
    rr_stop_prob = 0.05
    iterations = 40
    # to_mask = True
    tensorboard = False
    tensorboard_freq = 10
    win_size = 100
    n2n = Noise2Noise(parse_params, trainable=False)
    n2n.load_model(
        '/home/roironen/Path_Recycling/code/projects/1D_NN_output/mc-0929/n2n-epoch4470-0.00722.pt')
    n2n.model.train(False)
    device = torch.device("cuda") if n2n.use_cuda else torch.device("cpu")
    print(f"Device={device}")
    err = []
    recon_loss = []
    recon_loss_HR = []
    nn_err = []
    nn_recon_loss = []
    nn_recon_loss_HR = []
    nn_time_list = []
    nn_hr_err = []
    nn_hr_recon_loss = []
    nn_hr_time_list = []
    nn_input_grads = []
    nn_output_grads = []
    noisy_time_list = []
    err_Np_factor = []
    recon_loss_Np_factor = []
    Np_factor_time_list = []
    grad_HR_list = []
    noisy_cloud_profile = []
    Np_factor_cloud_profile = []
    nn_cloud_profile = []
    table_noisy = []
    table_clean = []
    table_nn = []
    table_noisy_loss = []
    table_clean_loss = []
    table_nn_loss = []
    for ind, file in enumerate(files):
        with open(glob.glob(os.path.join(file, 'cloud_info.pkl'))[0], 'rb') as f:
            x = pickle.load(f)
        shape = x['shape']
        beta_cloud = x['beta_gt']
        beta_cloud = beta_cloud.astype(float_reg)
        beta_max = beta_cloud.max()

        # Declerations
        grid = Grid(bbox, beta_cloud.shape)
        volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
        beta_gt = np.copy(beta_cloud)
        pytorch_scene_rr = PytorchSceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob,N_batches= 10, device=device)
        scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)

        for i in range(N_render_gt):
            cuda_paths = scene_rr.build_paths_list(Np_gt)
            cuda_paths_a = pytorch_scene_rr.build_paths_list(int(Np_gt/10))

            if i == 0:
                I_gt = scene_rr.render(cuda_paths)
                I_gt_a = pytorch_scene_rr.render(cuda_paths)
            else:
                I_gt += scene_rr.render(cuda_paths)
        I_gt /= N_render_gt
        cuda_paths = None
        cuda_paths_a = None
        max_val = np.max(I_gt, axis=(1, 2))
        # visual.plot_images(I_gt, "GT")
        # plt.show()
        del (cuda_paths)
        print("Calculating Cloud Mask")
        # cloud_mask = scene_rr.space_curving(I_gt, image_threshold=image_threshold, hit_threshold=hit_threshold, spp=spp)
        cloud_mask = beta_cloud > 0.01
        # mask_grader(cloud_mask, beta_gt>0.01, beta_gt)
        scene_rr.set_cloud_mask(cloud_mask)
        pytorch_scene_rr.set_cloud_mask(cloud_mask)
        alpha = 0.9
        beta1 = 0.9
        beta2 = 0.999
        start_iter = -1
        scaling_factor = 1.5


        ps = ps_max
        r = 1 / np.sqrt(scaling_factor)
        n = int(np.ceil(np.log(ps / ps_max) / np.log(r)))

        I_gts = [I_gt]
        pss = [ps_max]
        I_gt = I_gts[0]
        ps = pss[0]
        print(pss)
        # scene_rr.upscale_cameras(ps)

        # grad_norm = None
        non_min_couter = 0
        next_phase = False
        min_loss = 1  #
        Np_factors = [1, 1, 10]
        max_factor = np.max(Np_factors)
        for use_nn, Np_factor in zip([True, False, False], Np_factors):
            to_torch = False
            # Initialize trained model
            # optimizer = SGD(volume,step_size)
            optimizer = MomentumSGD(volume, step_size, alpha, beta_max, beta_max)
            # optimizer = ADAM(volume,step_size, beta1, beta2,start_iter , beta_max, beta_max, 1)

            # Initialization
            beta_init = np.zeros_like(beta_cloud)
            beta_init[volume.cloud_mask] = 2
            volume.set_beta_cloud(beta_init)
            beta_opt = volume.beta_cloud
            loss = 1
            # start_loop = time()
            grad = None
            loss_list = []
            rel_list = []
            time_list = []
            total_time = 0
            loss_HR_list = []
            best_err = np.inf
            grad_list = []
            nn_grad_list = []
            grad_HR = []
            for it in range(iterations):
                print(f"\niter {it}")
                abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
                max_dist = np.max(abs_dist)
                rel_dist1 = relative_distance(beta_cloud, beta_opt)
                if best_err > rel_dist1:
                    best_err = rel_dist1
                    best_cloud = np.squeeze(np.copy(beta_opt))
                print(
                    f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={int(Np * Np_factor):.2e}, ps={ps}, time={total_time}")

                start = time()

                if use_nn and it >= 0:
                    to_torch = True

                print("RESAMPLING PATHS ")
                if to_torch:
                    start1 = time()
                    pytorch_scene_rr.init_cuda_param(int(Np * Np_factor))
                    end = time()
                    print(f"A took: {end - start1}")

                    start1 = time()
                    cuda_paths = pytorch_scene_rr.build_paths_list(int(Np * Np_factor),to_print=True)
                    end = time()
                    print(f"B took: {end - start1}")

                    start1 = time()
                    I_opt, total_grad = pytorch_scene_rr.render(cuda_paths, I_gt=I_gt, to_torch=to_torch, to_print=True)
                    end = time()
                    print(f"C took: {end - start1}")

                    total_grad *= (ps * ps)
                    #
                    # for repeat in range(repeats):
                    #
                    #     grad_curr *= (ps * ps)
                    #     I_opt += I_curr
                    #     total_grad[repeat,cloud_mask.ravel()] = grad_curr.view(-1)[cloud_mask.ravel()]
                    # I_opt /= 10
                    total_grad = total_grad.view(-1)[None]
                    print(f"NORM {np.linalg.norm(total_grad[0].reshape(10,32).mean(dim=0).view(-1).cpu().detach().numpy(), ord=2)}")

                else:
                    start1 = time()
                    scene_rr.init_cuda_param(int(Np * Np_factor))
                    end = time()
                    print(f"A took: {end - start1}")

                    start1 = time()
                    cuda_paths = scene_rr.build_paths_list(int(Np * Np_factor))
                    end = time()
                    print(f"B took: {end - start1}")

                    start1 = time()
                    I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=I_gt)
                    total_grad *= (ps * ps)
                    print(f"C took: {end - start1}")

                    print(f"NORM {np.linalg.norm(total_grad.ravel(), ord=2)}")
                if Np_factor != max_factor:
                    vis_time = time()
                    cuda_paths = scene_rr.build_paths_list(int(Np * max_factor))
                    I_HR, g = scene_rr.render(cuda_paths, I_gt=I_gt)
                    dif_HR = (I_HR - I_gt).reshape(1, 1, 1, N_cams, *scene_rr.pixels_shape)
                    loss_HR = 0.5 * np.sum(dif_HR * dif_HR)
                    grad_HR.append(g)
                    vis_time = time() - vis_time
                else:
                    loss_HR = None
                    vis_time = 0
                end = time()
                print(f"gradient took: {end - start}")

                dif = (I_opt - I_gt).reshape(1, 1, 1, N_cams, *scene_rr.pixels_shape)
                # grad_norm = np.linalg.norm(total_grad)
                start1 = time()
                if use_nn and it>=0:
                    # mean = torch.mean(total_grad)
                    # std = torch.std(total_grad)
                    # total_grad = (total_grad-mean)/std
                    scale = 1e-4#torch.linalg.norm(total_grad.view(-1), ord=2) / torch.sum(total_grad != 0)
                    # grad_norm = torch.linalg.norm(total_grad.view(-1), ord=2)
                    total_grad /= scale
                    total_grad = n2n.model(total_grad)
                    # total_grad = total_grad * std + mean
                    total_grad *= scale
                    # total_grad /= torch.linalg.norm(total_grad.view(-1), ord=2)
                    # total_grad *= grad_norm
                    total_grad = torch.squeeze(total_grad).to("cpu").detach().numpy().reshape(1,1,32)
                    total_grad[np.logical_not(cloud_mask)] = 0
                    print(f"NORM AFTER {np.linalg.norm(total_grad.ravel(), ord=2)}")

                    print("NN took:",time()-start1)
                optimizer.step(total_grad)
                time_list.append(time() - start - vis_time)
                total_time += time_list[-1]
                loss = 0.5 * np.sum(dif * dif)
                loss_list.append(loss)
                rel_list.append(rel_dist1)
                if loss_HR:
                    loss_HR_list.append(loss_HR)
                if total_time > 300:
                    break
            if use_nn:
                nn_err.append(np.array(rel_list))
                nn_recon_loss.append(np.array(loss_list))
                nn_recon_loss_HR.append(np.array(loss_HR_list))
                nn_time_list.append(time_list)
                nn_input_grads.append(nn_grad_list)
                nn_output_grads.append(grad_list)
                grad_HR_list.append(grad_HR)

                nn_cloud_profile = best_cloud


            else:
                if Np_factor == 1:
                    err.append(np.array(rel_list))
                    recon_loss.append(np.array(loss_list))
                    recon_loss_HR.append(np.array(loss_HR_list))
                    noisy_time_list.append(time_list)
                    noisy_cloud_profile = best_cloud
                else:
                    err_Np_factor.append(np.array(rel_list))
                    recon_loss_Np_factor.append(np.array(loss_list))
                    Np_factor_time_list.append(time_list)
                    Np_factor_cloud_profile = best_cloud

            #################################
        print("MC rel_dist1={} | MC HR rel_dist1={} | NN + MC x10 rel_dist1={} ".format(err[-1][-1],
                                                                                        err_Np_factor[-1][-1],
                                                                                        nn_err[-1][-1]))
        plt.plot(err[-1])
        plt.plot(err_Np_factor[-1])
        plt.plot(nn_err[-1])

        plt.legend(['noisy', 'clean', 'NN denoiser x10'])
        plt.xlabel("Iterations")
        plt.ylabel("Epsilon - Relative error")
        plt.savefig(join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                         "eps_cloud{}_iterations".format(ind)), bbox_inches='tight')
        plt.close()

        plt.plot(np.cumsum(noisy_time_list[-1]), err[-1])
        plt.plot(np.cumsum(Np_factor_time_list[-1]), err_Np_factor[-1])
        plt.plot(np.cumsum(nn_time_list[-1]), nn_err[-1])

        plt.legend(['noisy', 'clean', 'NN denoiser x10'])
        plt.xlabel("Time [sec]")
        plt.ylabel("Epsilon - Relative error")
        plt.savefig(join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                         "eps_cloud{}_time".format(ind)), bbox_inches='tight')
        plt.close()
        ### Loss plots ###
        print("MC loss={} | MC HR loss={} | NN + MC x10 loss={} ".format(recon_loss[-1][-1],
                                                                         recon_loss_Np_factor[-1][-1],
                                                                         nn_recon_loss[-1][-1]))
        #################################
        plt.plot(recon_loss[-1])
        plt.plot(recon_loss_Np_factor[-1])
        plt.plot(nn_recon_loss[-1])

        plt.legend(['noisy', 'clean', 'NN denoiser x10'])
        plt.xlabel("Iterations")
        plt.ylabel("L2 loss")
        plt.yscale('log')
        plt.savefig(
            join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                 "loss_cloud{}_iterations".format(ind)),
            bbox_inches='tight')
        plt.close()

        plt.plot(np.cumsum(noisy_time_list[-1]), recon_loss[-1])
        plt.plot(np.cumsum(Np_factor_time_list[-1]), recon_loss_Np_factor[-1])
        plt.plot(np.cumsum(nn_time_list[-1]), nn_recon_loss[-1])

        plt.legend(['noisy', 'clean', 'NN denoiser x10'])
        plt.xlabel("Time [sec]")
        plt.ylabel("L2 loss")
        plt.yscale('log')
        plt.savefig(join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                         "loss_cloud{}_time".format(ind)), bbox_inches='tight')
        plt.close()

        #################################
        plt.plot(recon_loss_HR[-1])
        plt.plot(recon_loss_Np_factor[-1])
        plt.plot(nn_recon_loss_HR[-1])

        plt.legend(['noisy', 'clean', 'NN denoiser x10'])
        plt.xlabel("Iterations")
        plt.ylabel("MSE Images")  # L2 loss for High-Res. Rendering
        plt.yscale('log')
        plt.savefig(
            join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                 "loss_HR_cloud{}_iterations".format(ind)),
            bbox_inches='tight')
        plt.close()

        plt.plot(np.cumsum(noisy_time_list[-1]), recon_loss_HR[-1])
        plt.plot(np.cumsum(Np_factor_time_list[-1]), recon_loss_Np_factor[-1])
        plt.plot(np.cumsum(nn_time_list[-1]), nn_recon_loss_HR[-1])

        plt.legend(['noisy', 'clean', 'NN denoiser x10'])
        plt.xlabel("Time [sec]")
        plt.ylabel("L2 loss for High-Res. Rendering")
        plt.yscale('log')
        plt.savefig(join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                         "loss_HR_cloud{}_time".format(ind)), bbox_inches='tight')
        plt.close()

        #################################
        altitude = np.linspace(0, voxel_size_z * 32, 32)
        plt.plot(noisy_cloud_profile, altitude)
        plt.plot(Np_factor_cloud_profile, altitude)
        plt.plot(nn_cloud_profile, altitude)
        plt.plot(np.squeeze(beta_cloud), altitude)
        plt.legend(['noisy', 'clean', 'NN denoiser x10', 'GT'])
        plt.xlabel("Extinction [1/km]")
        plt.ylabel("Altitude")
        plt.savefig(
            join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                 "profile_cloud{}_iterations".format(ind)),
            bbox_inches='tight')
        plt.close()

        rand_iter = [1, 2, 3]  # [20,50,80]
        # for i in rand_iter:
        #     plt.plot(np.squeeze(np.mean(nn_input_grads[-1][i], 0)) * step_size, altitude, 'o')
        #     plt.plot(np.squeeze(grad_HR_list[-1][i]) * step_size, altitude, 'x')
        #     plt.plot(np.squeeze(nn_output_grads[-1][i]) * step_size, altitude, '^')
        #     plt.legend(['noisy', 'clean', 'NN output grad'])
        #     plt.xlabel("Extinction [1/km]")
        #     plt.ylabel("Altitude")
        #     plt.xscale('symlog')
        #     plt.savefig(
        #         join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
        #              "grads_cloud{}_iteration_{}".format(ind, i)),
        #         bbox_inches='tight')
        #     plt.close()
        ########################
        #####  Table #####
        # 'noisy', 'clean', 'NN denoiser x10']
        ########################
        noisy_arg = np.argmin(recon_loss_HR[-1])
        clean_arg = np.argmin(recon_loss_Np_factor[-1])
        nn_arg = np.argmin(nn_recon_loss_HR[-1])

        table_noisy.append(err[-1][noisy_arg])
        table_clean.append(err_Np_factor[-1][clean_arg])
        table_nn.append(nn_err[-1][nn_arg])

        table_noisy_loss.append(recon_loss_HR[-1][noisy_arg])
        table_clean_loss.append(recon_loss_Np_factor[-1][clean_arg])
        table_nn_loss.append(nn_recon_loss_HR[-1][nn_arg])

    print(f'Noisy best relative errors {table_noisy}, mean = {np.mean(table_noisy)}, std = {np.std(table_noisy)}')
    print(f'Clean best relative errors {table_clean}, mean = {np.mean(table_clean)}, std = {np.std(table_clean)}')
    print(f'NN best relative errors {table_nn}, mean = {np.mean(table_nn)}, std = {np.std(table_nn)}')
    results = {
        'err': table_noisy,
        'err_Np_factor': table_clean,
        'nn_err': table_nn,
        'recon_loss': table_noisy_loss,
        'recon_loss_Np_factor': table_clean_loss,
        'nn_recon_loss': table_nn_loss,
        'noisy_time_list': noisy_time_list,
        'Np_factor_time_list': Np_factor_time_list,
        'nn_time_list': nn_time_list
    }
    from scipy.io import savemat
    savemat(join("/home/roironen/Path_Recycling/code/test_results/1D_test_results1",
                 "TotalResults.mat"), results)
