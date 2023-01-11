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
from model_cloud.models import Cloud_Flows
from run_nerf_helpers import get_embedder
import torch.optim as optim
import torch.nn as nn


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
step_size = 5e9
# Ns = 15
rr_depth = 20
rr_stop_prob = 0.05
iterations = 200
# to_mask = True
tensorboard = False
tensorboard_freq = 10
win_size = 100

i_cloud = 0


def run_network(self, inputs, fn, is_rand, is_val, is_test, embed_fn, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    # outputs_flat, loss_entropy = self.batchify(fn, netchunk)(embedded, is_rand, is_val, is_test)
    outputs_flat, loss_entropy = fn(embedded, is_rand, is_val, is_test)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + list(outputs_flat.shape[-2:]))  # (B,N,K,4)

    return outputs, loss_entropy


def train(args, ground_truth, measurements):
    args.cuda = (args.ngpu > 0) and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    if args.cuda:
        # torch.cuda.manual_seed(args.seed)
        device = 'cuda'
    else:
        device = 'cpu'
    args.embed_fn, args.input_ch = get_embedder(args.multires, args.i_embed)
    args.skips = [args.netdepth / 2]
    args.device = device
    model = Cloud_Flows(args).to(device)
    # model = nn.DataParallel(model).to(device)
    # if not args.no_reload:
    #     ckpt_path = os.path.join(args.ft_path)
    #     print('Reloading from', ckpt_path)
    #     ckpt = torch.load(ckpt_path)
    grad_vars = list(model.parameters())
    network_query_fn = lambda inputs, network_fn, is_rand, is_val, is_test: run_network(inputs,
                                                                                             network_fn, is_rand,
                                                                                             is_val,
                                                                                             is_test,
                                                                                             embed_fn=args.embed_fn,
                                                                                             netchunk=args.netchunk_per_gpu * args.ngpu)

    gridx, gridy, gridz = ground_truth.lwc.grid.x, ground_truth.lwc.grid.y, ground_truth.lwc.grid.z
    gridx, gridy, gridz = np.meshgrid(torch.tensor(gridx),
                                      torch.tensor(gridy),
                                      torch.tensor(gridz))
    grid = torch.tensor(
        np.array([gridx.ravel()[optimizer.mask.data.ravel()], gridy.ravel()[optimizer.mask.data.ravel()],
                  gridz.ravel()[optimizer.mask.data.ravel()]])).T.to(device=device)


    error = nn.MSELoss()

    if args.opt == 'sgd':
        NNoptimizer = optim.SGD(params=grad_vars, lr=args.lr)
    elif args.opt == 'adam':
        NNoptimizer = torch.optim.Adam(params=grad_vars, lr=args.lr)
    elif args.opt == 'rmsprop':
        NNoptimizer = optim.RMSprop(params=grad_vars, lr=args.lr)
    elif args.opt == "lbfgs":
        NNoptimizer = torch.optim.LBFGS(params=grad_vars, lr=args.lr,
                                        max_iter=5, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09,
                                        history_size=100, line_search_fn=None)

    else:
        assert "Optimization strategy not defined"

    gd_relative_error_convergence = []
    relative_error_list = []
    NN_loss_list = []
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)
    # best_train_eps_convergence = []

    is_test = False
    print("===========Training===========")
    global is_rand
    is_rand = False
    soft = torch.nn.Softplus(1)
    for epoch in range(args.nEpochs):

        model.rand(grid.shape[0])

        for iter in range(200):
            NNoptimizer.zero_grad()
            x, loss_entropy = network_query_fn(grid, model, is_rand=is_rand, is_val=False,
                                               is_test=is_test)  # alpha_mean.shape (B,N,1)

            # x = soft(x)
            # x = F.softplus(x)

            loss_t = 0
            xx = torch.zeros(optimizer.mask.data.shape, device=x.device)
            xx.ravel()[optimizer.mask.data.ravel()] = torch.squeeze(x)
            y = forward_model(xx[None])

            loss = error(y, measurements)
            # loss = error(torch.squeeze(x), torch.tensor(ground_truth.lwc.data[optimizer.mask.data],dtype=torch.float,device=device))
            # loss_t += loss
            # output_np = xx.detach().cpu().numpy()

            # Calculating gradients
            loss.backward()
            # for p in model.parameters():
            #     print(p.grad)
            NN_loss_list.append(loss.mean().item())
            optimizer._loss = [loss.mean().item(), 0]

            optimizer._medium.set_state(np.squeeze(x.detach().cpu().numpy()))
            optimizer.callback(None)
            relative_error_list.append(optimizer.writer.epsilon)

            print('Epoch {}: Image Loss: {} Entropy Loss: {} relative_error: {}'.format(epoch, loss,
                                                                                        loss_entropy.item(),
                                                                                        relative_error_list[
                                                                                            -1]))
            NNoptimizer.step()

        if epoch % 5 == 0:
            path = os.path.join(optimizer.writer.dir,
                                '{}.tar'.format(optimizer.iteration))

            torch.save({
                'global_step': optimizer.iteration,
                'network_fn_state_dict': model.state_dict(),
                'optimizer_state_dict': NNoptimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # optimizer.callback(None)
        # f, axarr = plt.subplots(1, len(im), figsize=(16, 16))
        # for ax, image, image_gt in zip(axarr, im, y):
        #     ax.imshow(image.cpu())
        #     ax.axis('off')
        # plt.show()


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
    device = torch.device("cuda")
    print(f"Device={device}")

    output_dir = "3D_generated_data_noise2clean/cloud{}".format(i_cloud)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    cloud = {'file': file}
    # beta_cloud = np.reshape(beta_cloud,(1,1,32))
    beta_cloud = beta_cloud.astype(float_reg)
    # Declerations
    grid = Grid(bbox, beta_cloud.shape)
    volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
    beta_gt = np.copy(beta_cloud)
    scene_rr = PytorchSceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob, device=device)
    cloud['beta_gt'] = csr_matrix(beta_gt.ravel())#.tolist()
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
    # mask_grader(cloud_mask, beta_gt>0.01, beta_gt)
    scene_rr.set_cloud_mask(cloud_mask)
    # beta_scalar_init = scene_rr.find_best_initialization(beta_gt, I_gt,0,30,10,Np_gt,True)
    # cloud['cloud_mask'] = csr_matrix(cloud_mask.ravel())#.tolist()
    with open('{}/cloud_mask.pkl'.format(output_dir), 'wb') as f:
        print(f"saving cloud mask")
        pickle.dump(cloud_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
    alpha = 0.9
    beta1 = 0.9
    beta2 = 0.999
    # start_iter = -1
    scaling_factor = 1.5
    # optimizer = SGD(volume,step_size)
    # optimizer = MomentumSGD(volume, step_size, alpha, beta_max, beta_max)
    # optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_max, beta_max, 1)

    ps = ps_max
    # r = 1/np.sqrt(scaling_factor)
    # n = int(np.ceil(np.log(ps/ps_max)/np.log(r)))

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
    err = []
    for iter in range(iterations):
        # cloud['est_cloud{}'.format(iter)] = beta_opt.tolist()

        # if iter > start_iter:
        #     resample_freq = 1
        print(f"\niter {iter}")
        abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
        max_dist = np.max(abs_dist)
        rel_dist1 = relative_distance(beta_cloud, beta_opt)
        err.append(rel_dist1)
        with open('{}/beta_opt_it_{}_Np_{}.pkl'.format(output_dir,iter,int(Np)), 'wb') as f:
            pickle.dump(csr_matrix(beta_opt.ravel()), f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")
        grad_lr = np.zeros((10,*beta_cloud.shape))
        scene_rr.init_cuda_param(Np)
        print("RESAMPLING PATHS ")
        start = time()
        del(cuda_paths)
        cuda_paths = scene_rr.build_paths_list(int(Np))
        end = time()
        print(f"building path list took: {end - start}")
        # differentiable forward model
        start = time()
        I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=I_gt)
        total_grad *= (ps*ps)
        end = time()
        print(f"rendering took: {end-start}")

    i_cloud += 1
    plt.plot(err)
    plt.show()
    #################################
    altitude = np.linspace(0, voxel_size_z * beta_cloud.shape[2], beta_cloud.shape[2])
    b = np.squeeze(np.copy(beta_opt))
    b[b==0] = np.nan
    b = np.nanmean(b,axis=(0,1))

    b_gt = np.squeeze(np.copy(beta_cloud))
    b_gt[b_gt==0] = np.nan
    b_gt = np.nanmean(b_gt,axis=(0,1))
    plt.plot(b, altitude)
    plt.plot(b_gt, altitude)
    plt.legend(['Estimated', 'GT'])
    plt.xlabel("Extinction [1/km]")
    plt.ylabel("Altitude")
    plt.show()
