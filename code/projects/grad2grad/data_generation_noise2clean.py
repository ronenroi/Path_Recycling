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

N_cams = 9
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

i_cloud = 0
is_continue = False
for file in files:
    if i_cloud >= N_cloud:
        break
    beta_cloud = np.load(file)['lwc'] * 100
    beta_max = beta_cloud.max()
    output_dir = "generated_data_noise2clean/cloud{}".format(i_cloud)
    if is_continue:
        assert os.path.exists(output_dir)
    else:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    print(beta_cloud.mean())
    if beta_cloud.mean() < 0.1:
        continue
    cloud = {'file': file}
    beta_cloud = beta_cloud.astype(float_reg)
    # Declerations
    grid = Grid(bbox, beta_cloud.shape)
    volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
    beta_gt = np.copy(beta_cloud)
    scene_rr = SceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob)
    cloud['beta_gt'] = csr_matrix(beta_gt.ravel())#.tolist()
    cloud['shape'] = beta_gt.shape
    if is_continue:
        with open('{}/cloud_info.pkl'.format(output_dir), 'rb') as f:
            print(f"loading cloud info")
            x = pickle.load(f)
        assert cloud['file'] == x['file']
    else:
        with open('{}/cloud_info.pkl'.format(output_dir), 'wb') as f:
            print(f"saving cloud info")
            pickle.dump(cloud, f, protocol=pickle.HIGHEST_PROTOCOL)
    visual = Visual_wrapper(scene_rr)
    # visual.create_grid()
    # visual.plot_cameras()
    # visual.plot_medium()
    plt.show()

    if is_continue:
        with open('{}/I_gt.pkl'.format(output_dir), 'rb') as f:
            print(f"loading cloud iamges")
            x = pickle.load(f)
            I_gt = x.toarray().reshape(N_cams,ps_max,ps_max)
    else:
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
            pickle.dump(csr_matrix(cloud_mask.ravel()), f, protocol=pickle.HIGHEST_PROTOCOL)
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

        for i, Np_factor in enumerate([1, 10]):
            if is_continue:
                i += 6
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

            # dif = (I_opt - I_gt).reshape(1,1,1, N_cams, *scene_rr.pixels_shape)
            # grad_norm = np.linalg.norm(total_grad)
            # if i==1:
            #     name = "noise"
            # else:
            #     name = "clean"
            # cloud['total_grad_it_{}_{}'.format(iter, name)] = csr_matrix(total_grad.ravel())#.tolist()
            with open('{}/total_grad_it_{}_Np_{}_{}.pkl'.format(output_dir,iter,int(Np*Np_factor),i), 'wb') as f:
                print(f"saving cloud grad")
                pickle.dump(csr_matrix(total_grad.ravel()), f, protocol=pickle.HIGHEST_PROTOCOL)

        start = time()
        if is_continue:
            with open('{}/total_grad_it_{}_Np_{}_{}.pkl'.format(output_dir,iter,Np*10,5), 'rb') as f:
                x = pickle.load(f)
                total_grad = x.toarray().reshape(beta_gt.shape)
        optimizer.step(total_grad)
            # print("gradient step took:",time()-start)
        # loss = 0.5 * np.sum(dif * dif)
        # loss = 0.5 * np.sum(np.abs(dif))
        # if loss < min_loss:
        #     min_loss = loss
        #     non_min_couter = 0
        # else:
        #     non_min_couter += 1
        # print(f"loss = {loss}, grad_norm={grad_norm}, max_grad={np.max(total_grad)}")



        # cloud['loss{}'.format(iter)] = loss#.tolist()
        # cloud['rel_dist1{}'.format(iter)] = rel_dist1#.tolist()
        # # Writing scalar and images to tensorboard
        # if tensorboard and iter % tensorboard_freq == 0:
        #     tb.update(beta_opt, I_opt, loss, max_dist, rel_dist1, Np, iter, time()-start_loop)

    i_cloud += 1


