import matplotlib
# matplotlib.use('agg')
import os, time, copy
import argparse

import torch
import torchvision.transforms.functional

from model_cloud.models import Cloud_Flows

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import shutil
import os, sys
import socket

import numpy as np


my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
# from classes.scene_rr import *
from classes.scene_seed_roi import *
from classes.camera import *
# from classes.visual import *
# from utils import *
# from cuda_utils import *
import matplotlib.pyplot as plt
import pickle
# from classes.optimizer import *
from os.path import join
import glob
cuda.select_device(0)
import tensorboardX as tb
from collections import OrderedDict
from run_nerf_helpers import get_embedder
import scipy.io as sio
# from scipy.ndimage import rotate
from torchvision.transforms.functional import rotate
import PIL
import torch.fft
from scipy.ndimage.morphology import binary_erosion
DEFAULT_DATA_ROOT = '/wdata/roironen/Deploy/MC_code/projects/CF-NeRF/data/' \
if not socket.gethostname()=='visl-25u' else '/home/roironen/pytorch3d/projects/CT/data/'

def plot_hists_height(gt, cloud_list):
    mask = gt > 0
    mask = np.reshape(mask, (-1, mask.shape[-1])).sum(0) * 1.0
    mask[mask == 0] = np.nan
    mask /= mask
    mask *= np.arange(mask.shape[0])
    z_min_i = np.nanargmin(mask)
    z_max_i = np.nanargmax(mask)
    e = [(c - gt) for c in cloud_list]
    e = np.array(e)
    e_nan = np.array(e)
    e_nan[:, gt == 0] = None
    e_nan = e_nan.reshape((e.shape[0], -1, e.shape[-1]))
    e_nan = e_nan[:, :, z_min_i:z_max_i]
    voxel_mean = np.nanmean(e_nan, 0)
    voxel_std = np.nanstd(e_nan, 0)

    z = np.arange(z_min_i, z_max_i) * 0.04
    gradient = np.linspace(0, 1, len(z))
    fig, axs = plt.subplots(1, 3)
    for ax in axs:
        ax.set_box_aspect(1)
    fig_c, axs_c = plt.subplots(1, 3)
    for ax in axs_c:
        ax.set_box_aspect(1)

    axs[0].hist(voxel_mean.reshape(-1), 30)
    axs_c[0].hist(voxel_mean, 30, density=True, histtype='bar', stacked=True, label=z, color=plt.cm.jet(gradient))

    axs[1].hist(voxel_std.reshape(-1), 30)
    axs_c[1].hist(voxel_std, 30, density=True, histtype='bar', stacked=True, color=plt.cm.jet(gradient))

    norm = voxel_std / voxel_mean
    norm[np.abs(norm) > 10] = 10
    axs[2].hist(norm.reshape(-1), 30)
    axs_c[2].hist(norm, 30, density=True, histtype='bar', stacked=True, color=plt.cm.jet(gradient))

    fig_c.legend()
    plt.show()

# def plot_hists_erode(gt, cloud_list):
#     masks = []
#     masks.append(gt > 0)
#     while np.sum(masks[-1])>0:
#         prev_mask = masks[-1]
#         next_mask = binary_erosion(prev_mask)
#         masks.append(next_mask)
#
#     mask = np.reshape(mask, (-1, mask.shape[-1])).sum(0) * 1.0
#     mask[mask == 0] = np.nan
#     mask /= mask
#     mask *= np.arange(mask.shape[0])
#     z_min_i = np.nanargmin(mask)
#     z_max_i = np.nanargmax(mask)
#     e = [(c - gt) for c in cloud_list]
#     e = np.array(e)
#     e_nan = np.array(e)
#     e_nan[:, gt == 0] = None
#     e_nan = e_nan.reshape((e.shape[0], -1, e.shape[-1]))
#     e_nan = e_nan[:, :, z_min_i:z_max_i]
#     voxel_mean = np.nanmean(e_nan, 0)
#     voxel_std = np.nanstd(e_nan, 0)
#
#     z = np.arange(z_min_i, z_max_i) * 0.04
#     gradient = np.linspace(0, 1, len(z))
#     fig, axs = plt.subplots(1, 3)
#     for ax in axs:
#         ax.set_box_aspect(1)
#     fig_c, axs_c = plt.subplots(1, 3)
#     for ax in axs_c:
#         ax.set_box_aspect(1)
#
#     axs[0].hist(voxel_mean.reshape(-1), 30)
#     axs_c[0].hist(voxel_mean, 30, density=True, histtype='bar', stacked=True, label=z, color=plt.cm.jet(gradient))
#
#     axs[1].hist(voxel_std.reshape(-1), 30)
#     axs_c[1].hist(voxel_std, 30, density=True, histtype='bar', stacked=True, color=plt.cm.jet(gradient))
#
#     norm = voxel_std / voxel_mean
#     norm[np.abs(norm) > 10] = 10
#     axs[2].hist(norm.reshape(-1), 30)
#     axs_c[2].hist(norm, 30, density=True, histtype='bar', stacked=True, color=plt.cm.jet(gradient))
#
#     fig_c.legend()
#     plt.show()


def plot_mask(cloud):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    nx,ny, nz = cloud.shape
    # set the colors of each object
    x,y,z = np.linspace(0,(nx-1)*0.05,nx), np.linspace(0,(ny-1)*0.05,ny), np.linspace(0,(nz-1)*0.04,nz)
    yy, xx, zz = np.meshgrid(y, x, z,)

    mask = cloud>0
    # medium[cloud<0.1] = 0
    colors = plt.cm.gray(mask)
    xx = xx[mask]
    yy = yy[mask]
    zz = zz[mask]
    ax.scatter(xx, yy, zz, alpha=0.7)
    plt.xlim([0,(nx-1)*0.05])
    plt.ylim([0,(ny-1)*0.05])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_zlim([0.6,(nz-1)*0.04])

    plt.show()
if __name__ == "__main__":
    x = np.load(
        '/home/roironen/Path_Recycling/code/projects/CF-NeRF/logs/CF_CloudCT.15-Feb-2023-10:55:19/results.np.npy',
        allow_pickle=True).item()
    gt = x['gt_clouds']
    cloud_list = x['est_clouds']
    plot_hists_height(gt, cloud_list)
    mask = gt > 0
    mean_cloud = np.mean(cloud_list,0)
    OP = np.cumsum(2*mean_cloud[:,:,::-1],-1)[:,:,::-1]
    OP = OP[mask]
    # mask = np.copy(scene_rr.volume.cloud_mask)

    plot_hists_height(gt, cloud_list, mask)

    # e = [(c-gt)[mask] for c in cloud_list]
    # e = np.array(e)
    # voxel_std = np.std(e,0)
    # voxel_mean = np.mean(e, 0)
    # plt.hist(voxel_std, 30)
    # plt.show()
    # plt.hist(voxel_mean, 30)
    # plt.show()
    # norm = voxel_std/voxel_mean
    # norm[np.abs(norm)>10] = 10
    # plt.hist(norm[norm<10], 30)
    # plt.show()

    # fig2, ax2 = plt.subplots(nrows=1, ncols=1)  # two axes on figure
    # ax2.set_aspect('equal', adjustable='box')
    colors = np.array(
        ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan",
         "magenta"])

    gt_flat = gt[mask]
    i = np.random.permutation(gt_flat.size)[:len(colors)]
    for c in cloud_list:
        c = c[mask]
        plt.scatter(gt_flat[i], c[i], c=colors)
    plt.plot([0, 60], [0, 60], c='r', ls='--')
    plt.xlabel('GT')
    plt.ylabel('Estimated')
    plt.show()

    i = np.random.permutation(gt_flat.size)[:len(colors)]
    e = [c[mask] for c in cloud_list]
    e = np.array(e)
    voxel_std = np.std(e, 0)
    voxel_mean = np.mean(e, 0)
    plt.errorbar(gt_flat[i], voxel_mean[i], yerr=voxel_std[i], fmt="o", ecolor=colors)
    plt.scatter(gt_flat[i], voxel_mean[i], c=colors)

    plt.plot([0, 30], [0, 30], c='r', ls='--')
    plt.xlabel('GT')
    plt.ylabel('Estimated')
    plt.show()

    i = np.random.permutation(gt_flat.size)[:200]

    e = [c[mask] for c in cloud_list]
    e = np.array(e)
    voxel_std = np.std(e, 0)
    voxel_mean = np.mean(e, 0)
    plt.errorbar(gt_flat[i], voxel_mean[i], yerr=voxel_std[i], fmt="o")
    plt.scatter(gt_flat[i], voxel_mean[i])

    plt.plot([0, 60], [0, 60], c='r', ls='--')
    plt.xlabel('GT')
    plt.ylabel('Estimated')
    plt.show()

    plt.plot(gt[mask], voxel_std, 'o')
    plt.xlabel('GT voxel')
    plt.ylabel('Estimated voxel std')
    plt.show()

    OP -= OP.min()
    OP /= OP.max()
    plt.scatter(voxel_mean, voxel_std,c=plt.cm.jet(OP),alpha=0.8)
    plt.xlabel('Estimated voxel mean')
    plt.ylabel('Estimated voxel std')
    plt.colorbar()
    plt.show()




    # plot_hists_erode(gt, cloud_list)

    # plt.hist(norm[norm<10], 30)
    # plt.show()



