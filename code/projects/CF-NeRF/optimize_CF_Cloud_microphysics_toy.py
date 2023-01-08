import matplotlib
matplotlib.use('agg')
import os, time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil
import json

import torch

import shdom
from shdom.shdom_nn import *
from scripts.optimize_extinction_lbfgs import OptimizationScript as ExtinctionOptimizationScript

# from .GlowSHDOM.glow3D import Glow3D
from model_cloud.models import Cloud_Flows
# from GlowSHDOM.lbfgs import *

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from run_nerf_helpers import get_embedder
import torch.nn.functional as F

class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self,save_path):
        self.save_path = save_path



    def plot_curve(self, relative_error_list, gd_relative_error_convergence, NN_loss_list):
        result_png_path = os.path.join(self.save_path, 'results_GD_objective.png')
        title = 'Results/ Gradient Descent Objective'
        dpi = 80
        width, height = 1200, 800
        legend_fontsize = 10
        scale_distance = 48.8
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)

        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Gradient Descent Objective', fontsize=16)

        y_axis = np.array(gd_relative_error_convergence)
        x_axis = np.arange(len(y_axis))  # epochs

        plt.semilogy(x_axis, y_axis, lw=2)


        if self.save_path is not None:
            fig.savefig(result_png_path, dpi=dpi, bbox_inches='tight')
            # print('---- save figure {} into {}'.format(title, result_png_path))
        plt.close(fig)
        #################
        result_png_path = os.path.join(self.save_path, 'results_relative_error.png')
        title = 'Results/ Relative Error'
        fig = plt.figure(figsize=figsize)

        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Relative Error', fontsize=16)

        y_axis = np.array(relative_error_list)
        x_axis = np.arange(len(y_axis))

        plt.plot(x_axis, y_axis, color='g', lw=2)

        if self.save_path is not None:
            fig.savefig(result_png_path, dpi=dpi, bbox_inches='tight')
            # print('---- save figure {} into {}'.format(title, result_png_path))
        plt.close(fig)
        #################
        result_png_path = os.path.join(self.save_path, 'results_nn_loss.png')

        title = 'Results/ NN loss'
        fig = plt.figure(figsize=figsize)

        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('NN loss', fontsize=16)

        y_axis = np.array(NN_loss_list)
        x_axis = np.arange(len(y_axis))  # epochs


        plt.semilogy(x_axis, y_axis, color='g',lw=2)

        if self.save_path is not None:
            fig.savefig(result_png_path, dpi=dpi, bbox_inches='tight')
            # print('---- save figure {} into {}'.format(title, result_png_path))
        plt.close(fig)




class OptimizationScript(ExtinctionOptimizationScript):
    """
    Optimize: Micro-physics
    ----------------------
    Estimate micro-physical properties based on multi-spectral radiance/polarization measurements.
    Note that for convergence a fine enough sampling of effective radii and variances should be pre-computed in the
    Mie tables used by the forward model. This is due to the linearization of the phase-function and it's derivatives.

    Measurements are simulated measurements using a forward rendering script
    (e.g. scripts/render_radiance_toa.py).

    For example usage see the README.md

    For information about the command line flags see:
      python scripts/optimize_microphysics_lbfgs.py --help

    Parameters
    ----------
    scatterer_name: str
        The name of the scatterer that will be optimized.
    """
    def __init__(self, scatterer_name='cloud'):
        super().__init__(scatterer_name)

    def medium_args(self, parser):
        """
        Add common medium arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--use_forward_lwc',
                            action='store_true',
                            help='Use the ground-truth LWC.')
        parser.add_argument('--use_forward_reff',
                                action='store_true',
                                help='Use the ground-truth effective radius.')
        parser.add_argument('--use_forward_veff',
                            action='store_true',
                            help='Use the ground-truth effective variance.')
        parser.add_argument('--const_lwc',
                            action='store_true',
                            help='Keep liquid water content constant at a specified value (not optimized).')
        parser.add_argument('--const_reff',
                            action='store_true',
                            help='Keep effective radius constant at a specified value (not optimized).')
        parser.add_argument('--const_veff',
                            action='store_true',
                            help='Keep effective variance constant at a specified value (not optimized).')
        parser.add_argument('--radiance_threshold',
                            default=[0.0175],
                            nargs='+',
                            type=np.float32,
                            help='(default value: %(default)s) Threshold for the radiance to create a cloud mask.'
                            'Threshold is either a scalar or a list of length of measurements.')
        parser.add_argument('--lwc_scaling',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for liquid water content estimation')
        parser.add_argument('--reff_scaling',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for effective radius estimation')
        parser.add_argument('--veff_scaling',
                            default=1.0,
                            type=np.float32,
                            help='(default value: %(default)s) Pre-conditioning scale factor for effective variance estimation')
        return parser


    def neuralnetwork_args(self, parser):
        """
        Add common medium arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--nEpochs', type=int, default=1000)
        # parser.add_argument('--GD_steps', type=int, default=1)
        # parser.add_argument('--weight_decay', '--wd', default=1e-20, type=float,
        #                     metavar='W', help='weight decay')
        parser.add_argument('--no-cuda', action='store_true')
        parser.add_argument('--opt', type=str, default='adam',
                            choices=('sgd', 'adam', 'rmsprop','lbfgs'))
        # parser.add_argument('--model', type=str, help='model directory')
        # parser.add_argument('--init_strategy', type=str, default='x_const',
        #                     choices=('x_zero', 'x_const', 'z_zero', 'random_fixed_norm', 'pseudo_inverse'))
        # parser.add_argument('--init_std', type=float, default=0.1)
        # parser.add_argument('--z_penalty_squared', action="store_true", help="use ||z|| if True else ||z||^2")

        # parser.add_argument('--config', is_config_file=True,
        #                     help='config file path')
        parser.add_argument("--expname", type=str,
                            help='experiment name')
        parser.add_argument("--dataname", type=str, default='leaves',
                            help='data name')
        parser.add_argument("--basedir", type=str, default='./logs/',
                            help='where to store ckpts and logs')


        # training options optimize_skip
        parser.add_argument("--is_train", action='store_true',
                            help='train or evaluate')
        parser.add_argument("--uniformsample", action='store_true',
                            help='use uniformsample points to train or not')
        parser.add_argument("--optimize_global", action='store_true',
                            help='optimize_global or not')
        parser.add_argument("--optimize_skip", type=int, default=2,
                            help='optimize_skip or not')
        parser.add_argument("--use_prior", action='store_true',
                            help='use_prior or not')
        parser.add_argument("--netdepth", type=int, default=4,
                            help='layers in network')
        parser.add_argument("--netwidth", type=int, default=32,
                            help='channels per layer')
        parser.add_argument("--netdepth_fine", type=int, default=4,
                            help='layers in fine network')
        parser.add_argument("--netwidth_fine", type=int, default=32,
                            help='channels per layer in fine network')

        parser.add_argument("--model", type=str, default=None,
                            help='model name')
        parser.add_argument("--N_rand", type=int, default=512,
                            help='batch size (number of random rays per gradient step)')
        parser.add_argument("--lr", type=float, default=1e-4,
                            help='learning rate')
        parser.add_argument("--lr_unc", type=float, default=5e-4,
                            help='learning rate')
        parser.add_argument("--lr_decay", type=int, default=250,
                            help='exponential learning rate decay (in 1000 steps)')
        parser.add_argument("--chunk", type=int, default=1024 * 8,
                            help='number of rays processed in parallel, decrease if running out of memory')
        parser.add_argument('--ngpu', type=int, default=1)
        parser.add_argument("--netchunk_per_gpu", type=int, default=1024 * 64,
                            help='number of pts sent through network in parallel, decrease if running out of memory')
        parser.add_argument("--no_batching", action='store_true',
                            help='only take random rays from 1 image at a time')
        parser.add_argument("--no_reload", action='store_true',
                            help='do not reload weights from saved ckpt')
        parser.add_argument("--ft_path", type=str, default=None,
                            help='specific weights npy file to reload for coarse network')

        # flow options
        parser.add_argument("--type_flows", type=str, default='no_flow',
                            choices=['planar', 'IAF', 'realnvp', 'glow', 'orthogonal', 'householder',
                                     'triangular', 'no_flow'],
                            help="""Type of flows to use, no flows can also be selected""")
        parser.add_argument("--n_flows", type=int, default=4,
                            help='num of flows in normalizing flows')
        parser.add_argument("--n_hidden", type=int, default=128,
                            help='channels per layer in normalizing flow network')
        parser.add_argument("--h_alpha_size", type=int, default=32,
                            help='dims of h_context to normalizing flow network')
        parser.add_argument("--h_rgb_size", type=int, default=64,
                            help='dims of h_context to normalizing flow network')
        parser.add_argument("--z_size", type=int, default=4,
                            help='dims of samples from base distribution of normalizing flow network')

        # rendering options
        parser.add_argument("--N_samples", type=int, default=64,
                            help='number of coarse samples per ray')
        parser.add_argument("--K_samples", type=int, default=1,
                            help='number of monto-carlo samples per points')
        parser.add_argument("--N_importance", type=int, default=0,
                            help='number of additional fine samples per ray')
        parser.add_argument("--perturb", type=float, default=1.,
                            help='set to 0. for no jitter, 1. for jitter')
        parser.add_argument("--use_viewdirs", action='store_true',
                            help='use full 5D input instead of 3D')
        parser.add_argument("--i_embed", type=int, default=0,
                            help='set 0 for default positional encoding, -1 for none')
        parser.add_argument("--multires", type=int, default=10,
                            help='log2 of max freq for positional encoding (3D location)')
        parser.add_argument("--multires_views", type=int, default=4,
                            help='log2 of max freq for positional encoding (2D direction)')
        parser.add_argument("--raw_noise_std", type=float, default=0.,
                            help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

        parser.add_argument("--render_only", action='store_true',
                            help='do not optimize, reload weights and render out render_poses path')
        parser.add_argument("--render_test", action='store_true',
                            help='render the test set instead of render_poses path')
        parser.add_argument("--render_factor", type=int, default=0,
                            help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

        # training options
        parser.add_argument("--beta1", type=float, default=0.,
                            help='beta for balancing entropy loss and nll loss')
        parser.add_argument("--beta_u", type=float, default=0.1,
                            help='beta_uniformsample for balancing entropy loss and nll loss')
        parser.add_argument("--beta_p", type=float, default=0.05,
                            help='beta_prior for balancing entropy loss and nll loss')
        parser.add_argument("--precrop_iters", type=int, default=0,
                            help='number of steps to train on central crops')
        parser.add_argument("--precrop_frac", type=float,
                            default=.5, help='fraction of img taken for central crops')

        parser.add_argument("--colmap_depth", action='store_true',
                            help='fraction of img taken for central crops')
        parser.add_argument("--depth_lambda", type=float, default=0.1,
                            help='fraction of img taken for central crops')

        # dataset options
        parser.add_argument("--dataset_type", type=str, default='llff',
                            help='options: llff / blender / deepvoxels')
        parser.add_argument("--testskip", type=int, default=8,
                            help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

        ## deepvoxels flags
        parser.add_argument("--shape", type=str, default='greek',
                            help='options : armchair / cube / greek / vase')

        ## blender flags
        parser.add_argument("--white_bkgd", action='store_true',
                            help='set to render synthetic data on a white bkgd (always use for dvoxels)')
        parser.add_argument("--half_res", action='store_true',
                            help='load blender synthetic data at 400x400 instead of 800x800')

        ## llff flags
        parser.add_argument("--factor", type=int, default=8,
                            help='downsample factor for LLFF images')
        parser.add_argument("--no_ndc", action='store_true',
                            help='do not use normalized device coordinates (set for non-forward facing scenes)')
        parser.add_argument("--lindisp", action='store_true',
                            help='sampling linearly in disparity rather than depth')
        parser.add_argument("--spherify", action='store_true',
                            help='set for spherical 360 scenes')
        parser.add_argument("--llffhold", type=int, default=8,
                            help='will take every 1/N images as LLFF test set, paper uses 8')

        # logging/saving options
        parser.add_argument("--i_print", type=int, default=100,
                            help='frequency of console printout and metric loggin')
        parser.add_argument("--i_img", type=int, default=1000,
                            help='frequency of tensorboard image logging')
        parser.add_argument("--i_weights", type=int, default=10000,
                            help='frequency of weight ckpt saving')
        parser.add_argument("--i_testset", type=int, default=10000000,
                            help='frequency of testset saving')
        parser.add_argument("--i_video", type=int, default=5000000,
                            help='frequency of render_poses video saving')

        # emsemble setting
        parser.add_argument("--index_ensembles", type=int, default=1,
                            help='num of networks in ensembles')
        parser.add_argument("--index_step", type=int, default=-1,
                            help='step of weights to load in ensembles')

        return parser

    def parse_arguments(self):
        """
        Handle all the argument parsing needed for this script.

        Returns
        -------
        args: arguments from argparse.ArgumentParser()
            Arguments required for this script.
        cloud_generator: a shdom.CloudGenerator object.
            Creates the cloudy medium. The loading of Mie tables takes place at this point.
        air_generator: a shdom.AirGenerator object
            Creates the scattering due to air molecules
        """
        parser = argparse.ArgumentParser()
        parser = self.optimization_args(parser)
        parser = self.medium_args(parser)
        parser = self.neuralnetwork_args(parser)

        # Additional arguments to the parser
        subparser = argparse.ArgumentParser(add_help=False)
        subparser.add_argument('--init')
        subparser.add_argument('--add_rayleigh', action='store_true')
        parser.add_argument('--init',
                            default='Homogeneous',
                            help='(default value: %(default)s) Name of the generator used to initialize the atmosphere. \
                                  for additional generator arguments: python scripts/optimize_extinction_lbgfs.py --generator GENERATOR --help. \
                                  See generate.py for more documentation.')
        parser.add_argument('--add_rayleigh',
                            action='store_true',
                            help='Overlay the atmosphere with (known) Rayleigh scattering due to air molecules. \
                                  Temperature profile is taken from AFGL measurements of summer mid-lat.')

        init = subparser.parse_known_args()[0].init
        add_rayleigh = subparser.parse_known_args()[0].add_rayleigh

        CloudGenerator = None
        if init:
            CloudGenerator = getattr(shdom.generate, init)
            parser = CloudGenerator.update_parser(parser)

        AirGenerator = None
        if add_rayleigh:
            AirGenerator = shdom.generate.AFGLSummerMidLatAir
            parser = AirGenerator.update_parser(parser)

        self.args = parser.parse_args()
        self.cloud_generator = CloudGenerator(self.args) if CloudGenerator is not None else None
        self.air_generator = AirGenerator(self.args) if AirGenerator is not None else None
        self.args.save = 'logs/CF_CloudCT.{}'.format( time.strftime("%d-%b-%Y-%H:%M:%S"))

    def get_medium_estimator(self, measurements, ground_truth):
        """
        Generate the medium estimator for optimization.

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired measurements.
        ground_truth: shdom.Scatterer


        Returns
        -------
        medium_estimator: shdom.MediumEstimator
            A medium estimator object which defines the optimized parameters.
        """
        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            lwc_grid = ground_truth.lwc.grid
            reff_grid = ground_truth.reff.grid
            veff_grid = ground_truth.reff.grid
        else:
            lwc_grid = reff_grid = veff_grid = self.cloud_generator.get_grid()
        grid = lwc_grid + reff_grid + veff_grid

        # Find a cloud mask for non-cloudy grid points
        if self.args.use_forward_mask:
            mask = ground_truth.get_mask(threshold=0.01)
        else:
            carver = shdom.SpaceCarver(measurements)
            mask = carver.carve(grid, agreement=0.9, thresholds=self.args.radiance_threshold)

        # Define micro-physical parameters: either optimize, keep constant at a specified value or use ground-truth
        if self.args.use_forward_lwc:
            lwc = ground_truth.lwc
        elif self.args.const_lwc:
            lwc = self.cloud_generator.get_lwc(lwc_grid)
        else:
            lwc = shdom.GridDataEstimator(self.cloud_generator.get_lwc(lwc_grid),
                                          min_bound=1e-5,
                                          max_bound=2.0,
                                          precondition_scale_factor=self.args.lwc_scaling)
        lwc.apply_mask(mask)

        if self.args.use_forward_reff:
            reff = ground_truth.reff
        elif self.args.const_reff:
            reff = self.cloud_generator.get_reff(reff_grid)
        else:
            reff = shdom.GridDataEstimator(self.cloud_generator.get_reff(reff_grid),
                                           min_bound=ground_truth.min_reff,
                                           max_bound=ground_truth.max_reff,
                                           precondition_scale_factor=self.args.reff_scaling)
        reff.apply_mask(mask)

        if self.args.use_forward_veff:
            veff = ground_truth.veff
        elif self.args.const_veff:
            veff = self.cloud_generator.get_veff(veff_grid)
        else:
            veff = shdom.GridDataEstimator(self.cloud_generator.get_veff(veff_grid),
                                           max_bound=ground_truth.max_veff,
                                           min_bound=ground_truth.min_veff,
                                           precondition_scale_factor=self.args.veff_scaling)
        veff.apply_mask(mask)

        # Define a MicrophysicalScattererEstimator object
        cloud_estimator = shdom.MicrophysicalScattererEstimator(ground_truth.mie, lwc, reff, veff)
        cloud_estimator.set_mask(mask)

        # Create a medium estimator object (optional Rayleigh scattering)
        medium_estimator = shdom.MediumEstimator(
            loss_type=self.args.loss_type,
            stokes_weights=self.args.stokes_weights
        )
        if self.args.add_rayleigh:
            air = self.air_generator.get_scatterer(cloud_estimator.wavelength)
            medium_estimator.set_grid(cloud_estimator.grid + air.grid)
            medium_estimator.add_scatterer(air, 'air')
        else:
            medium_estimator.set_grid(cloud_estimator.grid)

        medium_estimator.add_scatterer(cloud_estimator, self.scatterer_name)
        return medium_estimator, mask

    def get_monotonous_reff(self, old_reff, slope, reff0, z0=None):
        mask = old_reff.data > 0
        grid = old_reff.grid
        if z0 is None:
            z0 = grid.z[mask][0]

        Z = grid.z - z0
        Z[Z < 0] = 0
        reff_data = (slope * Z ** (1. / 3.)) + reff0

        reff_data[Z == 0] = 0
        reff_data[mask == 0] = 0
        return shdom.GridData(grid, reff_data)

    def get_monotonous_lwc(self, old_lwc, slope, z0=None):
        mask = old_lwc.data > 0
        mask_z = np.sum(mask,(0,1))>0
        grid = old_lwc.grid
        if z0 is None:
            z0 = grid.z[mask_z][0]

        Z = grid.z - z0
        Z[Z < 0] = 0
        lwc_profile = (slope * Z ) + 0.01
        lwc_data = np.tile(lwc_profile[np.newaxis, np.newaxis, :], (grid.nx, grid.ny, 1))

        lwc_data[mask==0] = 0
        return shdom.GridData(grid, lwc_data)

    def get_medium_array_estimator(self, measurements: shdom.DynamicMeasurements, ground_truth: shdom.DynamicScatterer):
        """
        Generate the medium estimator for optimization.

        Parameters
        ----------
        measurements: shdom.DynamicMeasurements
            The acquired measurements.
        ground_truth: shdom.DynamicScatterer


        Returns
        -------
        medium_estimator: shdom.MediumEstimator
            A medium estimator object which defines the optimized parameters.
        """

        num_of_mediums = self.args.K_samples
        time_list = measurements.time_list
        temporary_scatterer_list = ground_truth._temporary_scatterer_list


        wavelength = ground_truth.wavelength
        if not isinstance(wavelength, list):
            wavelength = [wavelength]
        # Define the grid for reconstruction
        if self.args.use_forward_grid:
            lwc_grid = []
            reff_grid = []
            veff_grid = []
            l_grid = []
            r_grid = []
            v_grid = []
            for temporary_scatterer in temporary_scatterer_list:
                l_grid.append(temporary_scatterer.scatterer.lwc.grid)
                r_grid.append(temporary_scatterer.scatterer.reff.grid)
                v_grid.append(temporary_scatterer.scatterer.veff.grid)

            l_grid = np.split(np.array(l_grid), num_of_mediums)
            r_grid = np.split(np.array(r_grid), num_of_mediums)
            v_grid = np.split(np.array(v_grid), num_of_mediums)
            for l, r, v in zip(l_grid, r_grid, v_grid):
                lwc_grid.append(np.sum(l))
                reff_grid.append(np.sum(r))
                veff_grid.append(np.sum(v))
            grid = lwc_grid[0]

        else:
            grid = self.cloud_generator.get_grid()
            grid = shdom.Grid(x=grid.x + temporary_scatterer_list[0].scatterer.lwc.grid.xmin, y=grid.y + temporary_scatterer_list[0].scatterer.lwc.grid.ymin, z=grid.z)
            lwc_grid = reff_grid = veff_grid = [grid]*num_of_mediums
        # if self.args.one_dim_reff:
        #     for i, grid in enumerate(reff_grid):
        #         reff_grid[i] = shdom.Grid(z=grid.z)

        # if self.args.use_forward_cloud_velocity:
        #     if ground_truth.num_scatterers > 1:
        #         cloud_velocity = ground_truth.get_velocity()[0]
        #     else:
        #         cloud_velocity = [0,0,0]
        #     cloud_velocity = np.array(cloud_velocity)*1000 #km/sec to m/sec
        # else:
        #     cloud_velocity = None

        # Find a cloud mask for non-cloudy grid points
        self.thr = 1e-3

        if self.args.use_forward_mask:
            mask_list = ground_truth.get_mask(threshold=self.thr)
            show_mask = 0
            if show_mask:
                a = (mask_list[0].data).astype(int)
                print(np.sum(a))
                shdom.cloud_plot(a)

        else:
            dynamic_carver = shdom.DynamicSpaceCarver(measurements)
            mask_list, dynamic_grid, cloud_velocity = dynamic_carver.carve(grid, agreement=0.9,
                                                                           time_list=measurements.time_list,
                                                                           thresholds=self.args.radiance_threshold,
                                                                           vx_max=0, vy_max=0,
                                                                           gt_velocity=0,
                                                                           verbose = False)
            show_mask = 1
            if show_mask:
                a = (mask_list[0].data).astype(int)
                b = ((ground_truth.get_mask(threshold=self.thr)[0].resample(dynamic_grid[0]).data)).astype(int)
                print(np.sum((a > b)))
                print(np.sum((a < b)))
                shdom.cloud_plot(a)
                shdom.cloud_plot(b)

        # Define micro-physical parameters: either optimize, keep constant at a specified value or use ground-truth
        if self.args.use_forward_lwc:
            lwc = ground_truth.get_lwc()[:num_of_mediums]#TODO use average
            for ind in range(len(lwc)):
                lwc[ind] = lwc[ind].resample(lwc_grid[ind])
        elif self.args.const_lwc:
            lwc = self.cloud_generator.get_lwc(lwc_grid)

        else:
            rr = []
            for m, r in zip(mask_list, lwc_grid):
                rr.append(self.get_monotonous_lwc(m.resample(r), 0.3))
            lwc = shdom.DynamicGridDataEstimator(rr,
                                          min_bound=1e-5,
                                          max_bound=2.0,
                                          precondition_scale_factor=self.args.lwc_scaling)
            # lwc = shdom.DynamicGridDataEstimator(self.cloud_generator.get_lwc(lwc_grid),
            #                               min_bound=1e-5,
            #                               max_bound=2.0,
            #                               precondition_scale_factor=self.args.lwc_scaling)
            lwc = lwc.dynamic_data


        if self.args.use_forward_reff:
            reff = ground_truth.get_reff()[:num_of_mediums] #TODO use average
            for ind in range(len(reff)):
                reff[ind] = reff[ind].resample(reff_grid[ind])

        elif self.args.const_reff:
            # reff = self.cloud_generator.get_reff(reff_grid)
            reff = []
            for m, r in zip(mask_list, reff_grid):
                reff.append(self.get_monotonous_reff(m.resample(r), 6.67, 3)) #cloud1
                # reff.append(self.get_monotonous_reff(m.resample(r), 7.71, 4, z0=reff_grid[0].z[18])) #cloud2

        else:
            rr=[]
            for m,r in zip(mask_list,reff_grid):
                rr.append (self.get_monotonous_reff(m.resample(r), 6.67, 3))#cloud1
                # rr.append (self.get_monotonous_reff(m.resample(r), 7.71, 4, z0=reff_grid[0].z[18]))#cloud2

            reff = shdom.DynamicGridDataEstimator(rr,
                                           min_bound=0.01,
                                           max_bound=35,
                                           precondition_scale_factor=self.args.reff_scaling)
            # reff = shdom.DynamicGridDataEstimator(self.cloud_generator.get_reff(reff_grid),
            #                                min_bound=0.01,
            #                                max_bound=35,
            #                                precondition_scale_factor=self.args.reff_scaling)
            reff = reff.dynamic_data

        if self.args.use_forward_veff:
            veff = ground_truth.get_veff()[:num_of_mediums] #TODO use average
            for ind in range(len(veff)):
                veff[ind] = veff[ind].resample(veff_grid[ind])
        elif self.args.const_veff:
            veff = self.cloud_generator.get_veff(veff_grid)
        else:
            veff = shdom.DynamicGridDataEstimator(self.cloud_generator.get_veff(veff_grid),
                                           min_bound=0.01,
                                           max_bound=0.3,
                                           precondition_scale_factor=self.args.veff_scaling)
            veff = veff.dynamic_data


        for lwc_i, reff_i, veff_i, mask in zip(lwc, reff, veff, mask_list):
            lwc_i.apply_mask(mask)
            reff_i.apply_mask(mask)
            veff_i.apply_mask(mask)

        # for l in lwc:
        #     l._data=(self.get_monotonous_lwc(mask_list[0], 0.3)).data
        # for r in reff:
        #     r._data = (self.get_monotonous_reff(r,6.67,3))._data
        #     a=r.data
        # r = r.resample(l.grid)
        # import matplotlib.pyplot as plt
        # shdom.cloud_plot(lwc_i.data)
        # plt.plot(reff_i.data)
        # plt.show()
        # Define a MicrophysicalScattererEstimator object
        kw_microphysical_scatterer = {"lwc": lwc, "reff": reff, "veff": veff}
        cloud_estimator = shdom.DynamicScattererEstimator(wavelength=wavelength, time_list=time_list, **kw_microphysical_scatterer)
        cloud_estimator.set_mask(mask_list)

        # Create a medium estimator object (optional Rayleigh scattering)

        air = self.air_generator.get_scatterer(cloud_estimator.wavelength)
        # sigma = {"lwc": self.args.sigma[0], 'reff': self.args.sigma[1]}
        sigma = {"lwc": 0, 'reff': 0}
        medium_estimator = shdom.DynamicMediumEstimator(cloud_estimator, air, sigma=sigma, loss_type=self.args.loss_type,
                                                        stokes_weights=self.args.stokes_weights, regularization_const=1)
        return medium_estimator, mask


    def load_forward_model(self, input_directory):
        """
        Load the ground-truth medium, rte_solver and measurements which define the forward model

        Parameters
        ----------
        input_directory: str
            The input directory where the forward model is saved

        Returns
        -------
        ground_truth: shdom.OpticalScatterer
            The ground truth scatterer
        rte_solver: shdom.RteSolverArray
            The rte solver with the numerical and scene parameters
        measurements: shdom.Measurements
            The acquired measurements
        """
        # Load forward model and measurements
        medium, rte_solver, measurements = shdom.load_forward_model(input_directory)

        # Get micro-physical medium ground-truth
        ground_truth = medium.get_scatterer(self.scatterer_name)
        return ground_truth, rte_solver, measurements

    def get_summary_writer(self, measurements, ground_truth):
        """
        Define a SummaryWriter object

        Parameters
        ----------
        measurements: shdom.Measurements object
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground-truth scatterer for monitoring

        Returns
        -------
        writer: shdom.SummaryWriter object
            A logger for the TensorboardX.
        """
        writer = None
        if self.args.log is not None:
            log_dir = os.path.join(self.args.input_dir, 'logs', self.args.log + '-' + time.strftime("%d-%b-%Y-%H:%M:%S"))
            writer = shdom.SummaryWriter(log_dir)
            writer.save_checkpoints(ckpt_period=-1)
            writer.monitor_loss()
            # writer.monitor_images(measurements=measurements, ckpt_period=-1)

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.4, parameters=['lwc'])
            writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=0.01))

        return writer

    def get_summary_writer_array(self, measurements, ground_truth):
        """
        Define a SummaryWriter object

        Parameters
        ----------
        measurements: shdom.Measurements object
            The acquired measurements.
        ground_truth: shdom.Scatterer
            The ground-truth scatterer for monitoring

        Returns
        -------
        writer: shdom.SummaryWriter object
            A logger for the TensorboardX.
        """
        writer = None
        if self.args.log is not None:
            log_dir = os.path.join(self.args.input_dir, 'logs', self.args.log + '-' + time.strftime("%d-%b-%Y-%H:%M:%S"))
            writer = shdom.DynamicSummaryWriter(log_dir)
            writer.save_checkpoints()
            writer.monitor_loss()
            # writer.monitor_images(measurements=measurements)

            # writer.monitor_state()

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            # writer.monitor_domain_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.8)
            # writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=self.thr))

            # self.save_args(log_dir)
        return writer


    def batchify(self, fn, chunk):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        if chunk is None:
            return fn

        def ret(inputs, is_val, is_test):
            for i in range(0, inputs.shape[0], chunk):
                # t0 = time.time()
                a, b = fn(inputs[i:i + chunk], is_val, is_test)
                # print('single forward time:',time.time()-t0)
                if i == 0:
                    A = a
                    B = b
                else:
                    A = torch.cat([A, a], dim=0)
                    B = torch.cat([B, b], dim=0)
            return A, B

        return ret

    def run_network(self, inputs, fn, is_rand, is_val, is_test, embed_fn, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat)

        # outputs_flat, loss_entropy = self.batchify(fn, netchunk)(embedded, is_rand, is_val, is_test)
        outputs_flat, loss_entropy = fn(embedded, is_rand, is_val, is_test)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + list(outputs_flat.shape[-2:]))  # (B,N,K,4)

        return outputs, loss_entropy

    def get_NNoptimizer(self, ground_truth, rte_solver, measurements):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """


        # Add noise (currently only supports AirMSPI noise model)
        if self.args.add_noise:
            noise = shdom.AirMSPINoise()
            measurements = noise.apply(measurements)

        # Initialize a Medium Estimator
        medium_estimator, mask = self.get_medium_estimator(measurements, ground_truth)

        # Initialize TensorboardX logger
        writer = self.get_summary_writer(measurements, ground_truth)

        optimizer = NNOptimizer(n_jobs=self.args.n_jobs)
        optimizer.set_measurements(measurements)
        optimizer.set_rte_solver(rte_solver)
        optimizer.set_medium_estimator(medium_estimator)
        optimizer.set_mask(mask)
        optimizer.set_writer(writer)

        # Reload previous state
        if self.args.reload_path is not None:
            optimizer.load_state(self.args.reload_path)
            # np.save('Rico_pyshdom', medium_estimator.scatterers['cloud'].lwc.data)
        return optimizer

    def get_NNoptimizer_array(self, ground_truth, rte_solver, measurements):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """


        # Add noise (currently only supports AirMSPI noise model)
        if self.args.add_noise:
            noise = shdom.AirMSPINoise()
            measurements = noise.apply(measurements)

        # Initialize a Medium Estimator
        medium_estimator, mask = self.get_medium_array_estimator(measurements, ground_truth)

        # Initialize TensorboardX logger
        writer = self.get_summary_writer_array(measurements, ground_truth)

        optimizer = NNOptimizer_array(n_jobs=self.args.n_jobs)
        optimizer.set_measurements(measurements)
        optimizer.set_dynamic_solver(rte_solver)
        optimizer.set_medium_estimator(medium_estimator)
        optimizer.set_mask(mask)
        optimizer.set_writer(writer)

        # Reload previous state
        if self.args.reload_path is not None:
            optimizer.load_state(self.args.reload_path)
            # np.save('Rico_pyshdom', medium_estimator.scatterers['cloud'].lwc.data)
        return optimizer

    def data2array(self, ground_truth, rte_solver, measurements):
        scatterer_array = shdom.DynamicScatterer()
        scatterer_array.generate_dynamic_scatterer(scatterer=ground_truth, time_list=[0]*self.args.K_samples,
                                                     scatterer_velocity_list=[0,0,0])
        rte_solver = shdom.DynamicRteSolver(scene_params=rte_solver._scene_parameters, numerical_params=rte_solver._numerical_parameters)
        projection = shdom.DynamicProjection([measurements.camera.projection]*self.args.K_samples)
        camera = shdom.DynamicCamera(measurements.camera.sensor, projection)
        measurements = shdom.DynamicMeasurements(camera=camera, images=measurements.images*self.args.K_samples, wavelength=measurements.wavelength,
                                                 time_list=[0]*self.args.K_samples)

        return scatterer_array, rte_solver, measurements

    def main(self):
        """
        Main optimization script
        """
        self.parse_arguments()

        ground_truth, rte_solver, measurements = self.load_forward_model(self.args.input_dir)
        if self.args.K_samples==0:
            local_optimizer = self.get_NNoptimizer(ground_truth, rte_solver, measurements)
        else:
            ground_truth_array, rte_solver, measurements = self.data2array(ground_truth, rte_solver, measurements)
            local_optimizer = self.get_NNoptimizer_array(ground_truth_array, rte_solver, measurements)
        self.train(local_optimizer, ground_truth, rte_solver, measurements)


        # Save optimizer state
        save_dir =  self.args.input_dir
        local_optimizer.save_state(os.path.join(save_dir, 'final_state.ckpt'))


    def train(self, optimizer, ground_truth, rte_solver, measurements):
        self.args.cuda = (self.args.ngpu > 0) and torch.cuda.is_available()
        # torch.manual_seed(self.args.seed)
        if self.args.cuda:
            # torch.cuda.manual_seed(self.args.seed)
            device = 'cuda'
        else:
            device = 'cpu'
        self.args.embed_fn, self.args.input_ch = get_embedder(self.args.multires, self.args.i_embed)
        self.args.skips = [self.args.netdepth / 2]
        self.args.device = device
        model = Cloud_Flows(self.args).to(device)
        # model = nn.DataParallel(model).to(device)
        # if not self.args.no_reload:
        #     ckpt_path = os.path.join(self.args.ft_path)
        #     print('Reloading from', ckpt_path)
        #     ckpt = torch.load(ckpt_path)
        grad_vars = list(model.parameters())
        network_query_fn = lambda inputs, network_fn, is_rand,is_val,  is_test: self.run_network(inputs,
                                                                                             network_fn, is_rand,is_val,
                                                                                             is_test,
                                                                                             embed_fn=self.args.embed_fn,
                                                                                             netchunk=self.args.netchunk_per_gpu * self.args.ngpu)
        # f_x = ForwardModel(self.args.input_dir)
        # x_gt = ground_truth.lwc.data
        gridx,gridy,gridz = ground_truth.lwc.grid.x,ground_truth.lwc.grid.y,ground_truth.lwc.grid.z
        gridx, gridy, gridz = np.meshgrid(torch.tensor(gridx),
                                                torch.tensor(gridy),
                                                torch.tensor(gridz))
        grid = torch.tensor(np.array([gridx.ravel()[optimizer.mask.data.ravel()], gridy.ravel()[optimizer.mask.data.ravel()],
                                     gridz.ravel()[optimizer.mask.data.ravel()]])).T.to(device=device)
        # grid *=1000
        # def forward_model(x):
        #     f = nn.AvgPool3d(2, 2)
        #     return f(x)

        def forward_model(x):
            y_down = int(x.shape[-2]/2)
            up = nn.AvgPool3d(2,2)
            # down = nn.AvgPool3d(1, 1)
            x_down = x[...,:y_down,:].flatten()
            x_up = up(x[...,y_down:,:]).flatten()
            x_new = torch.cat([x_down,x_up])
            return x_new

        measurements = forward_model(torch.tensor(ground_truth.lwc.data[None],dtype=torch.float,device=device))
        # L2 Loss
        # if self.args.K_samples==0:
        #     loss_shdom = LossSHDOM().apply
        # else:
        #     loss_shdom = LossSHDOM_array().apply


        error = nn.MSELoss()

        if self.args.opt == 'sgd':
            NNoptimizer = optim.SGD(params=grad_vars, lr=self.args.lr)
        elif self.args.opt == 'adam':
            NNoptimizer = torch.optim.Adam(params=grad_vars, lr=self.args.lr)
        elif self.args.opt == 'rmsprop':
            NNoptimizer = optim.RMSprop(params=grad_vars, lr=self.args.lr)
        elif self.args.opt == "lbfgs":
            NNoptimizer = torch.optim.LBFGS(params=grad_vars, lr=self.args.lr,
                                            max_iter=5, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09,
                                            history_size=100, line_search_fn=None)

        else:
            assert "Optimization strategy not defined"

        gd_relative_error_convergence = []
        relative_error_list = []
        NN_loss_list = []
        if os.path.exists(self.args.save):
            shutil.rmtree(self.args.save)
        os.makedirs(self.args.save, exist_ok=True)
        recorder = RecorderMeter(self.args.save)
        # best_train_eps_convergence = []

        is_test=False
        print("===========Training===========")
        global is_rand
        is_rand = False
        soft = torch.nn.Softplus(1)
        for epoch in range(self.args.nEpochs):
            def closure():
                # Clear gradients
                if torch.is_grad_enabled():
                    NNoptimizer.zero_grad()
                # NNoptimizer.zero_grad(set_to_none=False)
                x, loss_entropy = network_query_fn(grid, model, is_rand=is_rand, is_val=False,
                                                   is_test=is_test)  # alpha_mean.shape (B,N,1)

                # x = soft(x)
                # x = F.softplus(x)
                # mask = 1 - optimizer._mask.data
                # x2 = x.detach().clone()
                # x = torch.squeeze(torch.clamp(x, 0, 2))
                # with torch.no_grad():
                #     mask_zero_clouds = torch.sum(x, 0) > 0.1
                # x = x[:, mask_zero_clouds]
                # x[:,:,mask] = 0

                # print(torch.norm(x - x2))
                # global err
                loss_t = 0
                # loss_t += loss_entropy
                # loss = torch.tensor(-1.0)
                # if True in mask_zero_clouds:
                xx = torch.zeros(optimizer.mask.data.shape,device=x.device)
                xx.ravel()[optimizer.mask.data.ravel()] = torch.squeeze(x)
                y = forward_model(xx[None])

                loss = error(y,measurements)
                loss_t += loss
                output_np = xx.detach().cpu().numpy()

                # err = error(loss_t)
                # fx = torch.tensor(np.array(optimizer.images), dtype=torch.float, device=x.device)
                # global im
                # im = fx

                # Calculating gradients
                if loss_t.requires_grad:
                    loss_t.backward(retain_graph=False)
                NN_loss_list.append(loss.mean().item())
                optimizer._loss = [loss.mean().item(),0]

                optimizer._medium.set_state(np.squeeze(x.detach().cpu().numpy()))
                # optimizer._images = images
                optimizer.callback(None)
                relative_error_list.append(optimizer.writer.epsilon)
                    # np.sum(np.sum(np.abs(ground_truth.lwc.data - output_np)) / np.sum(np.abs(ground_truth.lwc.data))))

                print('Epoch {}: Image Loss: {} Entropy Loss: {} relative_error: {}'.format(epoch, loss, loss_entropy.item(), relative_error_list[-1]))
                # recorder.plot_curve(relative_error_list, gd_relative_error_convergence, NN_loss_list)
                return loss_t
            # # Clear gradients
            # NNoptimizer.zero_grad()
            # x, loss_entropy = network_query_fn(grid, model, is_val=False,
            #                                    is_test=is_test)  # alpha_mean.shape (B,N,1)
            # x = F.softplus(x)
            # # mask = 1 - optimizer._mask.data
            # # x2 = x.detach().clone()
            # x = torch.squeeze(torch.clamp(x, 0, 2))
            # with torch.no_grad():
            #     mask_zero_clouds = torch.sum(x, 0) > 0.1
            # x = x[:, mask_zero_clouds]
            # # x[:,:,mask] = 0
            #
            # # print(torch.norm(x - x2))
            # # global err
            # loss_t = 0
            # # loss_t += loss_entropy
            # loss = torch.tensor(-1.0)
            # if True in mask_zero_clouds:
            #     loss = loss_shdom(x, optimizer)
            #     loss_t += loss.mean()
            #     output_np = x.detach().cpu().numpy()
            #
            # err = error(loss_t)
            # # fx = torch.tensor(np.array(optimizer.images), dtype=torch.float, device=x.device)
            # # global im
            # # im = fx
            #
            # # Calculating gradients
            # err.backward(retain_graph=False)
            # NN_loss_list.append(loss.mean().item())
            #
            # optimizer.callback(None)
            # relative_error_list.append(
            #     optimizer.writer.epsilon)  # np.sum(np.sum(np.abs(x_gt - output_np)) / np.sum(np.abs(x_gt)))
            #
            # print(
            #     'Epoch {}: Image Loss: {} Entropy Loss: {} relative_error: {}'.format(epoch, loss, loss_entropy.item(),
            #                                                                           relative_error_list[-1]))
            # recorder.plot_curve(relative_error_list, gd_relative_error_convergence, NN_loss_list)
            # Update parameters
            model.rand(grid.shape[0])

            for iter in range(200):
                if self.args.opt == "lbfgs":
                    NNoptimizer.step(closure)
                else:
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





if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()



