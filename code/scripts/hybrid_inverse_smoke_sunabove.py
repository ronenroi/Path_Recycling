import os, sys
my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from classes.scene import *
from classes.scene_hybrid_gpu import *
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
from tqdm import tqdm
cuda.select_device(0)


########################
# Atmosphere parameters#
########################
sun_angles = np.array([180, 0]) * (np.pi / 180)

#####################
# Volume parameters #
#####################
# construct betas
beta_cloud = loadmat(join("data", "smoke.mat"))["data"] * 50
beta_cloud = np.ascontiguousarray(np.rot90(beta_cloud, axes=(2,1)))
beta_cloud =np.roll(beta_cloud, axis=0, shift=-15)
# beta_cloud = beta_cloud.T
# beta_cloud = loadmat(join("data", "rico2.mat"))["vol"]
beta_cloud = beta_cloud.astype(float_reg)
# beta_cloud *= (127/beta_cloud.max())
# Grid parameters #
# bounding box
voxel_size_x = 0.02
voxel_size_y = 0.02
voxel_size_z = 0.02
edge_x = voxel_size_x * beta_cloud.shape[0]
edge_y = voxel_size_y * beta_cloud.shape[1]
edge_z = voxel_size_z * beta_cloud.shape[2]
bbox = np.array([[0, edge_x],
                 [0, edge_y],
                 [0, edge_z]])


print(beta_cloud.shape)
print(bbox)

beta_air = 0.004
# w0_air = 1.0 #0.912
w0_air = 0.912
# w0_cloud = 0.8 #0.9
w0_cloud = 0.9
g_cloud = 0.5

# Declerations
grid = Grid(bbox, beta_cloud.shape)
volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
beta_gt = np.copy(beta_cloud)
# phase_function = (UniformPhaseFunction)
#######################
# Cameras declaration #
#######################




ps_max = 200
pixels = np.array([ps_max, ps_max])
N_cams = 9
R = 1.5 * edge_z
#
height_factor = 1.5
focal_length = 50e-3
sensor_size = np.array((47e-3, 47e-3)) / height_factor
volume_center = (bbox[:, 1] - bbox[:, 0]) / 1.65
cameras = []
R = height_factor * edge_z

for cam_ind in range(N_cams):
    theta = np.pi/2
    phi = (-(N_cams//2) + cam_ind) * 40
    phi_rad = phi * (np.pi/180)
    t = R * theta_phi_to_direction(theta,phi_rad) + volume_center
    t[2] -= 0.55
    euler_angles = np.array((90, 0, phi-90))
    camera = Camera(t, euler_angles, focal_length, sensor_size, pixels)
    cameras.append(camera)


# Simulation parameters
Np_gt = int(5e7)
Np_max = int(5e7)
Np = int(1e6)
resample_freq = 10
step_size = 5e5
# Ns = 15
Ns = 15
iterations = 10000000
to_mask = True
tensorboard = True
tensorboard_freq = 10
beta_max = beta_cloud.max()
win_size = 100


scene_hybrid = SceneHybridGpu(volume, cameras, sun_angles, g_cloud, Ns)

visual = Visual_wrapper(scene_hybrid)
# visual.create_grid()
# visual.plot_cameras()
# visual.plot_medium()
plt.show()
cuda_paths = scene_hybrid.build_paths_list(Np_gt, Ns)
I_gt = scene_hybrid.render(cuda_paths)
del(cuda_paths)
cuda_paths = None
max_val = np.max(I_gt, axis=(1,2))
visual.plot_images(I_gt, "GT")
plt.show()

print("Calculating Cloud Mask")
cloud_mask = scene_hybrid.space_curving(I_gt, image_threshold=0.9, hit_threshold=0.9, spp=1000)
mask_grader(cloud_mask, beta_gt>0.1, beta_gt)
scene_hybrid.set_cloud_mask(cloud_mask)



scene_hybrid.init_cuda_param(Np)
alpha = 0.9
beta1 = 0.9
beta2 = 0.999
start_iter = 500
scaling_factor = 1.5
# optimizer = SGD(volume,step_size)
beta_mean = np.mean(beta_cloud[volume.cloud_mask])
# optimizer = MomentumSGD(volume, step_size, alpha, beta_mean, beta_max)
optimizer = ADAM(volume,step_size, beta1, beta2, start_iter, beta_mean, beta_max, 1)

ps = 30
# n = int(np.log(Np_gt/Np)/np.log(1.5))
r = 1/np.sqrt(scaling_factor)
n = int(np.ceil(np.log(ps/ps_max)/np.log(r)))

I_gts = [I_gt]
pss = [ps_max]
print("creating I_gt pyramid...")
for iter in tqdm(range(n)):
    if iter < n-1:
        ps_temp = int(ps_max * r**iter)
        temp = ps_temp/ps_max
    else:
        temp = ps/ps_max
        ps_temp = ps
    I_temp = zoom(I_gt, (1,temp, temp), order=1)
    I_temp *= 1/(temp*temp) # light correction factor
    I_gts.insert(0,I_temp)
    pss.insert(0, ps_temp)
#
I_gt = I_gts[0]
ps = pss[0]
print(pss)
scene_hybrid.upscale_cameras(ps)
if tensorboard:
    tb = TensorBoardWrapper(I_gt, beta_gt)
    cp_wrapper = CheckpointWrapper(scene_hybrid, optimizer, Np_gt, Np, Ns, resample_freq, step_size, iterations,
                            tensorboard_freq, tb.train_id)
    tb.add_scene_text(str(cp_wrapper))
    pickle.dump(cp_wrapper, open(join(tb.folder,"data","checkpoint_loader"), "wb"))
    print("Checkpoint wrapper has been saved")




# grad_norm = None
non_min_couter = 0
next_phase = False
min_loss = 1#
upscaling_counter = 0
# photon_scale = (ps/ps_gt)**2
# cuda_paths = scene_lowmem.build_paths_list(int(Np_gt*photon_scale), Ns)
# I_gt = scene_lowmem.render(cuda_paths)
# I_gt = I_gts[0]
tb.update_gt(I_gt)
# Initialization
beta_init = np.zeros_like(beta_cloud)
beta_init[volume.cloud_mask] = np.mean(beta_gt[beta_gt>0.1])
# beta_init[volume.cloud_mask] = 0
volume.set_beta_cloud(beta_init)
beta_opt = volume.beta_cloud
loss = 1
for iter in range(iterations):
    # if iter > start_iter:
    #     resample_freq = 1
    # print(f"\niter {iter}")
    abs_dist = np.abs(beta_cloud[cloud_mask] - beta_opt[cloud_mask])
    max_dist = np.max(abs_dist)
    rel_dist1 = relative_distance(beta_cloud, beta_opt)

    print(f"rel_dist1={rel_dist1}, loss={loss} max_dist={max_dist}, Np={Np:.2e}, ps={ps} counter={non_min_couter}")

    if iter % resample_freq == 0:
        if non_min_couter >= win_size and iter > start_iter:
            if Np < Np_max :
                Np = int(Np * scaling_factor)
                resample_freq = 30
                non_min_couter = 0
                # step_size *= scaling_factor
                if Np > Np_max:
                    Np = Np_max

            if ps < ps_max:
                upscaling_counter += 1
                # photon_scale = (ps / ps_gt) ** 2
                ps = pss[upscaling_counter]
                scene_hybrid.upscale_cameras(ps)
                # volume.beta_cloud = beta_gt
                # cuda_paths = scene_lowmem.build_paths_list(int(Np_gt*photon_scale), Ns)
                I_gt = I_gts[upscaling_counter]
                # I_gt = scene_lowmem.render(cuda_paths)
                tb.update_gt(I_gt)
                # volume.beta_cloud = beta_opt
        print("RESAMPLING PATHS ")
        start = time()
        del(cuda_paths)
        cuda_paths = scene_hybrid.build_paths_list(Np, Ns)
        end = time()
        print(f"building path list took: {end - start}")
    # differentiable forward model
    start = time()
    I_opt, total_grad = scene_hybrid.render(cuda_paths, I_gt=I_gt)
    total_grad *= (ps*ps)
    end = time()
    print(f"rendering took: {end-start}")


    dif = (I_opt - I_gt).reshape(1,1,1, N_cams, *scene_hybrid.pixels_shape)
    grad_norm = np.linalg.norm(total_grad)

    # updating beta
    # beta_opt -= step_size*total_grad
    # beta_opt[beta_opt >= beta_max] = beta_mean
    # beta_opt[beta_opt < 0] = 0
    # loss calculation
    start = time()
    optimizer.step(total_grad)
    # print("gradient step took:",time()-start)
    loss = 0.5 * np.sum(dif * dif)
    # loss = 0.5 * np.sum(np.abs(dif))
    if loss < min_loss:
        min_loss = loss
        non_min_couter = 0
    else:
        non_min_couter += 1
    # print(f"loss = {loss}, grad_norm={grad_norm}, max_grad={np.max(total_grad)}")

    # Writing scalar and images to tensorboard
    if tensorboard and iter % tensorboard_freq == 0:
        tb.update(beta_opt, I_opt, loss, max_dist, rel_dist1, Np, iter)




