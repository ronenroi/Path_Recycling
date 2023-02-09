import matplotlib
# matplotlib.use('agg')
import os, time, copy
import argparse

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


DEFAULT_DATA_ROOT = '/wdata/roironen/Deploy/MC_code/projects/CF-NeRF/data/' \
if not socket.gethostname()=='visl-25u' else '/home/roironen/pytorch3d/projects/CT/data/'

class NNOptimizer(object):

    def __init__(self, n_jobs=1, init_solution=True):
        self._medium = None
        self._rte_solver = None
        self._measurements = None
        self._writer = None
        self._images = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._init_solution = init_solution
        self._mask = None

    def set_measurements(self, measurements):
        """
        Set the measurements (data-fit constraints)

        Parameters
        ----------
        measurements: shdom.Measurements
            A measurements object storing the acquired images and sensor geometry
        """
        self._measurements = measurements

    def set_medium_estimator(self, medium_estimator):
        """
        Set the MediumEstimator for the optimizer.

        Parameters
        ----------
        medium_estimator: shdom.MediumEstimator
            The MediumEstimator
        """
        self._medium = medium_estimator

    def set_mask(self, mask):
        """
        Set the Mask for the optimizer.

        Parameters
        ----------
        mask: numpy array
            The Mask
        """
        self._mask = mask

    def set_writer(self, writer):
        """
        Set a log writer to upload summaries into tensorboard.

        Parameters
        ----------
        writer: shdom.SummaryWriter
            Wrapper for the tensorboardX summary writer.
        """
        self._writer = writer
        if writer is not None:
            self._writer.attach_optimizer(self)


    def objective_fun(self, state):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        if self.iteration == 0:
            self.init_optimizer()
        # self._iteration += 1
        self.set_state(state)
        gradient, loss, images = self.medium.compute_gradient(
            rte_solvers=self.rte_solver,
            measurements=self.measurements,
            n_jobs=self.n_jobs
        )
        # import matplotlib.pyplot as plt
        # f, axarr = plt.subplots(1, len(images), figsize=(16, 16))
        # for ax, image in zip(axarr, images):
        #     ax.imshow(image)
        #     ax.invert_xaxis()
        #     ax.invert_yaxis()
        #     ax.axis('off')
        # plt.show()
        self._loss = loss
        self._images = images
        return loss, gradient

    def init_optimizer(self):
        """
        Initialize the optimizer.
        This means:
          1. Setting the RteSolver medium
          2. Initializing a solution
          3. Computing the direct solar flux derivatives
          4. Counting the number of unknown parameters
        """

        assert self.rte_solver.num_solvers == self.measurements.num_channels == self.medium.num_wavelengths, \
            'RteSolver has {} solvers, Measurements have {} channels and Medium has {} wavelengths'.format(
                self.rte_solver.num_solvers, self.measurements.num_channels, self.medium.num_wavelengths)

        self.rte_solver.set_medium(self.medium)
        self.rte_solver.init_solution()
        self.medium.compute_direct_derivative(self.rte_solver)
        self._num_parameters = self.medium.num_parameters

    def callback(self,state):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector
        """
        self._iteration += 1

        # Writer callback functions
        if self.writer is not None:
            flush= False if self.iteration>1 else True
            for callbackfn, kwargs in zip(self.writer.callback_fns, self.writer.kwargs):
                time_passed = time.time() - kwargs['ckpt_time']
                kwargs['ckpt_time'] = time.time()
                callbackfn(kwargs)
                if time_passed > kwargs['ckpt_period']:
                    flush = True
            if flush:
                self.writer.tf_writer.close()
    # def get_bounds(self):
    #     """
    #     Retrieve the bounds for every parameter from the MediumEstimator (used by scipy.minimize)
    #
    #     Returns
    #     -------
    #     bounds: list of tuples
    #         The lower and upper bound of each parameter
    #     """
    #     return self.medium.get_bounds()


    def get_state(self):
        """
        Retrieve MediumEstimator state

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        return self.medium#.get_state()

    def set_state(self, state):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        self.medium = state


    def save_state(self, path):
        """
        Save Optimizer state to file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.get_state(), -1))
        file.close()

    def load_state(self, path):
        """
        Load Optimizer from file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'rb')
        data = file.read()
        file.close()
        state = pickle.loads(data)
        self.set_state(state)

    @property
    def rte_solver(self):
        return self._rte_solver

    @property
    def medium(self):
        return self._medium

    @property
    def measurements(self):
        return self._measurements

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def writer(self):
        return self._writer

    @property
    def iteration(self):
        return self._iteration

    @property
    def mask(self):
        return self._mask

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def loss(self):
        return self._loss

    @property
    def images(self):
        return self._images


class ArraySummaryWriter(object):
    """
    A wrapper for tensorboardX summarywriter with some basic summary writing implementation.
    This wrapper enables logging of images, error measures and loss with pre-determined temporal intervals into tensorboard.

    To view the summary of this run (and comparisons to all subdirectories):
        tensorboard --logdir LOGDIR

    Parameters
    ----------
    log_dir: str
        The directory where the log will be saved
    """

    def __init__(self, log_dir=None):
        self._dir = log_dir
        self._tf_writer = tb.SummaryWriter(log_dir,flush_secs=120,max_queue=10) if log_dir is not None else None
        self._ground_truth_parameters = None
        self._callback_fns = []
        self._kwargs = []
        self.epsilon = np.nan
        self._optimizer = None

    def add_callback_fn(self, callback_fn, kwargs=None):
        """
        Add a callback function to the callback function list

        Parameters
        ----------
        callback_fn: bound method
            A callback function to push into the list
        kwargs: dict, optional
            A dictionary with optional keyword arguments for the callback function
        """
        self._callback_fns.append(callback_fn)
        self._kwargs.append(kwargs)

    def attach_optimizer(self, optimizer):
        """
        Attach the optimizer

        Parameters
        ----------
        optimizer: shdom.Optimizer
            The optimizer that the writer will report for
        """
        self._optimizer = optimizer

    def monitor_loss(self, ckpt_period=-1):
        """
        Monitor the loss.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'loss',
        }
        self.add_callback_fn(self.loss_cbfn, kwargs)

    def monitor_time_smoothness(self, ckpt_period=-1):
        """
        Monitor the time smoothness.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'time_smoothness'
        }
        self.add_callback_fn(self.time_smoothness_cbfn, kwargs)

    def save_checkpoints(self, ckpt_period=-1):
        """
        Save a checkpoint of the Optimizer

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time()
        }
        self.add_callback_fn(self.save_ckpt_cbfn, kwargs)

    def monitor_state(self, ckpt_period=-1):
        """
        Monitor the state of the optimization.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        self.states = []
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time()
        }
        self.add_callback_fn(self.state_cbfn, kwargs)

    def monitor_shdom_iterations(self, ckpt_period=-1):
        """Monitor the number of SHDOM forward iterations.

        Parameters
        ----------
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'shdom iterations'
        }
        self.add_callback_fn(self.shdom_iterations_cbfn, kwargs)

    def monitor_scatterer_error(self, estimator_name, ground_truth, ckpt_period=-1):
        """
        Monitor relative and overall mass error (epsilon, delta) as defined at:
          Amit Aides et al, "Multi sky-view 3D aerosol distribution recovery".

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['{}/delta/{} at time{}', '{}/epsilon/{} at time{}']
        }
        self.add_callback_fn(self.scatterer_error_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth.beta_cloud.copy()
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth.beta_cloud.copy()})

    def monitor_scatter_plot(self, estimator_name, ground_truth, dilute_percent=0.4, ckpt_period=-1, parameters='all'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/scatter_plot/{}{}',
            'percent': dilute_percent,
            'parameters': parameters
        }
        self.add_callback_fn(self.scatter_plot_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth.beta_cloud.copy()
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth.beta_cloud.copy()})

    # def monitor_horizontal_mean(self, estimator_name, ground_truth, ground_truth_mask=None, ckpt_period=-1):
    #     """
    #     Monitor horizontally averaged quantities and com        return writerpare to ground truth over iterations.
    #
    #     Parameters
    #     ----------
    #     estimator_name: str
    #         The name of the scatterer to monitor
    #     ground_truth: shdom.Scatterer
    #         The ground truth medium.
    #     ground_truth_mask: shdom.GridData
    #         The ground-truth mask of the estimator
    #     ckpt_period: float
    #        time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
    #     """
    #     kwargs = {
    #         'ckpt_period': ckpt_period,
    #         'ckpt_time': time.time(),
    #         'title': '{}/horizontal_mean/{}{}',
    #         'mask': ground_truth_mask
    #     }
    #
    #     self.add_callback_fn(self.horizontal_mean_cbfn, kwargs)
    #     if hasattr(self, '_ground_truth'):
    #         self._ground_truth[estimator_name] = ground_truth.beta_cloud.copy()
    #     else:
    #         self._ground_truth = OrderedDict({estimator_name: ground_truth.beta_cloud.copy()})

    def monitor_domain_mean(self, estimator_name, ground_truth, ckpt_period=-1):
        """
        Monitor domain mean and compare to ground truth over iterations.

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': '{}/mean/{}'
        }
        self.add_callback_fn(self.domain_mean_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._ground_truth[estimator_name] = ground_truth
        else:
            self._ground_truth = OrderedDict({estimator_name: ground_truth})

    def monitor_images(self, measurements, ckpt_period=-1):
        """
        Monitor the synthetic images and compare to the acquired images

        Parameters
        ----------
        measurements: shdom.Measurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        acquired_images = measurements
        num_images = acquired_images.shape[0]

        vmax = np.max(acquired_images)


        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Retrieval/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax
        }
        self.add_callback_fn(self.estimated_images_cbfn, kwargs)

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Diff/view{}'.format(view) for view in range(num_images)],
            'acquired_images': acquired_images,
            'vmax': vmax
        }
        self.add_callback_fn(self.diff_images_cbfn, kwargs)

        kwargs = {
            'vmax': vmax
        }
        acq_titles = ['Acquired/view{}'.format(view) for view in range(num_images)]
        self.write_image_list(0, acquired_images, acq_titles, vmax=kwargs['vmax'])
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'loss/normalized loss image {}',
            'acquired_images': acquired_images,
            'vmax': vmax
        }
        self.add_callback_fn(self.loss_norm_cbfn, kwargs)

    def save_ckpt_cbfn(self, kwargs=None):
        """
        Callback function that saves checkpoints .

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        timestr = time.strftime("%H%M%S")
        path = os.path.join(self.tf_writer.logdir, timestr + '.ckpt')
        self.optimizer.save_state(path)

    def loss_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        if isinstance(self.optimizer.loss,list):
            self.tf_writer.add_scalars(kwargs['title'], {
                kwargs['title']: sum(self.optimizer.loss),
                'Data loss': self.optimizer.loss[0],
                'Entropy loss': self.optimizer.loss[1],
            }
                , self.optimizer.iteration)
        else:
            self.tf_writer.add_scalars(kwargs['title'], {
                kwargs['title']: self.optimizer.loss,
            }
                                       , self.optimizer.iteration)
    # def loss_cbfn(self, kwargs):
    #     """
    #     Callback function that is called (every optimizer iteration) for loss monitoring.
    #
    #     Parameters
    #     ----------
    #     kwargs: dict,
    #         keyword arguments
    #     """
    #     loss = np.sum([np.sum((im1 - im2)**2) for im1, im2 in zip(kwargs['acquired_images'], self.optimizer.images)])
    #     self.tf_writer.add_scalar(kwargs['Data term loss'], diff, self.optimizer.iteration)

    def loss_norm_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """

        losses_norm = [np.sum((im1 - im2).ravel()**2) ** 0.5 / (im1.max()+1e-12) / im1.size for im1, im2 in zip(kwargs['acquired_images'], self.optimizer.images)]
        for i, loss in enumerate(losses_norm):
            self.tf_writer.add_scalars('loss/normalized loss', {kwargs['title'].format(i): loss}, self.optimizer.iteration)

    def time_smoothness_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        extinctions=[]
        for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
            for estimator_temporary_scatterer in est_scatterer.temporary_scatterer_estimator_list:
                extinctions.append(estimator_temporary_scatterer.scatterer.extinction.data)
        err=0
        for ind, (extinction_i, extinction_j) in enumerate(zip(extinctions[:-1], extinctions[1:])):
            # err += np.linalg.norm((extinction_i - extinction_j).reshape(-1, 1), ord=2)
            err = np.linalg.norm((extinction_i - extinction_j).reshape(-1, 1), ord=2)
        # self.tf_writer.add_scalar(kwargs['title'], err, self.optimizer.iteration)
            self.tf_writer.add_scalars(
                main_tag=kwargs['title'],
                tag_scalar_dict={'{}'.format(ind): err},
                global_step=self.optimizer.iteration
            )

    def state_cbfn(self, kwargs=None):
        """
        Callback function that is called for state monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        state = np.empty(shape=(0), dtype=np.float64)
        for estimator in self.optimizer.medium.estimators.values():
            for param in estimator.estimators.values():
                state = np.concatenate((state, param.get_state() / param.precondition_scale_factor))
        self.states.append(state)

    def diff_images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        diff = [np.abs(im1 - im2) for im1, im2 in zip(kwargs['acquired_images'], self.optimizer.images)]
        self.write_image_list(self.optimizer.iteration, diff, kwargs['title'], kwargs['vmax'])

    def estimated_images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.write_image_list(self.optimizer.iteration, self.optimizer.images, kwargs['title'], kwargs['vmax'])

    def shdom_iterations_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for shdom iteration monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalar(kwargs['title'], self.optimizer.rte_solver.num_iterations, self.optimizer.iteration)

    def scatterer_error_cbfn(self, kwargs):
        """
        Callback function for monitoring parameter error measures.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for name, gt_cloud in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.copy()
            eps_list =[]
            delta_list=[]
            if len(est_scatterer.shape)==3:
                gt_clouds = gt_cloud[None].copy()
                est_scatterers = est_scatterer[None]

            colud_num = 0
            for gt_param, est_param in zip(gt_clouds, est_scatterers):
                est_param = est_param.flatten()
                gt_param = gt_param.flatten()
                delta = (np.linalg.norm(est_param, 1) - np.linalg.norm(gt_param, 1)) / np.linalg.norm(gt_param, 1)
                epsilon = np.linalg.norm((est_param - gt_param), 1) / np.linalg.norm(gt_param, 1)
                self.tf_writer.add_scalar(kwargs['title'][0].format(name, 'ext', colud_num), delta,
                                          self.optimizer.iteration)
                self.tf_writer.add_scalar(kwargs['title'][1].format(name, 'ext', colud_num), epsilon,
                                          self.optimizer.iteration)
                eps_list.append(epsilon)
                delta_list.append(delta)
                colud_num +=1
            self.tf_writer.add_scalar(
                'eps', np.mean(eps_list),
                self.optimizer.iteration)
            self.tf_writer.add_scalar(
                'delta', np.mean(np.abs(delta_list)),
                self.optimizer.iteration)
            self.epsilon = np.mean(eps_list)

    def domain_mean_cbfn(self, kwargs):
        """
        Callback function for monitoring domain averages of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """

        for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
            parm_scatterer = est_scatterer.temporary_scatterer_estimator_list[0]
            for parameter_name, parameter in parm_scatterer.scatterer.estimators.items():
                for ind, gt_temporary_scatterer in \
                        enumerate(gt_dynamic_scatterer.temporary_scatterer_list):
                    gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                    gt_param_mean = gt_param.data.mean()
                    self.tf_writer.add_scalars(
                        main_tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name),
                        tag_scalar_dict={'true{}'.format(ind): gt_param_mean},
                        global_step=self.optimizer.iteration
                    )
                for ind, estimator_temporary_scatterer in \
                        enumerate( est_scatterer.temporary_scatterer_estimator_list):
                    est_param = getattr(estimator_temporary_scatterer.scatterer, parameter_name).resample(gt_param.grid)
                    est_param_mean = est_param.data.mean()
                    self.tf_writer.add_scalars(
                        main_tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name),
                        tag_scalar_dict={'estimated{}'.format(ind): est_param_mean},
                        global_step=self.optimizer.iteration
                    )

    # def horizontal_mean_cbfn(self, kwargs):
    #     """
    #     Callback function for monitoring horizontal averages of parameters.
    #
    #     Parameters
    #     ----------
    #     kwargs: dict,
    #         keyword arguments
    #     """
    #
    #     for dynamic_scatterer_name, gt_dynamic_scatterer in self._ground_truth.items():
    #         est_scatterer = self.optimizer.medium.get_scatterer(dynamic_scatterer_name)
    #         parm_scatterer = est_scatterer.temporary_scatterer_estimator_list[0]
    #         for parameter_name, parameter in parm_scatterer.scatterer.estimators.items():
    #             est_temporary_scatterer_estimator_list = []
    #             est_temporary_scatterer_estimator_list +=( est_scatterer.temporary_scatterer_estimator_list)
    #             if len(est_temporary_scatterer_estimator_list)<len(gt_dynamic_scatterer.temporary_scatterer_list):
    #                 est_temporary_scatterer_estimator_list *= len(gt_dynamic_scatterer.temporary_scatterer_list)
    #             for ind, (gt_temporary_scatterer, estimator_temporary_scatterer) in \
    #                     enumerate(zip(gt_dynamic_scatterer.temporary_scatterer_list,
    #                             est_temporary_scatterer_estimator_list)):
    #                 common_grid = estimator_temporary_scatterer.scatterer.grid + gt_temporary_scatterer.scatterer.grid
    #                 a = estimator_temporary_scatterer.scatterer.get_mask(threshold=0.0).resample(common_grid, method='nearest')
    #                 b = gt_temporary_scatterer.scatterer.get_mask(threshold=0.0).resample(common_grid, method='nearest')
    #                 common_mask = shdom.GridData(data=np.bitwise_or(a.data, b.data), grid=common_grid)
    #                 gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
    #                 est_parameter = getattr(estimator_temporary_scatterer.scatterer, parameter_name)
    #
    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore", category=RuntimeWarning)
    #
    #                     est_parameter_masked = copy.deepcopy(est_parameter).resample(common_grid)
    #                     est_parameter_masked.apply_mask(common_mask)
    #                     est_param = est_parameter_masked.data
    #                     est_param[np.bitwise_not(common_mask.data)] = np.nan
    #                     est_param_mean = np.nan_to_num(np.nanmean(est_param, axis=(0, 1)))
    #
    #                     gt_param_masked = copy.deepcopy(gt_param).resample(common_grid)
    #                     gt_param_masked.apply_mask(common_mask)
    #                     gt_param = gt_param_masked.data
    #                     gt_param[np.bitwise_not(common_mask.data)] = np.nan
    #                     gt_param_mean = np.nan_to_num(np.nanmean(gt_param, axis=(0, 1)))
    #                     # if parameter.type == 'Homogeneous' or parameter.type == '1D':
    #                     #     est_param = getattr(estimator_temporary_scatterer.scatterer, parameter_name).resample(
    #                     #         gt_param.grid)
    #                     #     est_param_mean = est_param.data
    #                     # else:
    #                     #     est_param = getattr(estimator_temporary_scatterer.scatterer, parameter_name).resample(
    #                     #         gt_param.grid)
    #                     #     est_param_data = copy.copy(est_param.data)
    #                     #     est_param_data[(estimator_temporary_scatterer.scatterer.mask.resample(gt_param.grid).data == False)] = np.nan
    #                     #     est_param_mean = np.nan_to_num(np.nanmean(est_param_data, axis=(0, 1)))
    #                     # if gt_param.type == 'Homogeneous' or gt_param.type == '1D':
    #                     #         gt_param_mean = gt_param.data
    #                     # else:
    #                     #
    #                     #     gt_param_data = copy.copy(gt_param.data)
    #                     #     if kwargs['mask']:
    #                     #         gt_param_data[kwargs['mask'][ind].data == False] = np.nan
    #                     #         # gt_param[estimator_temporary_scatterer.scatterer.mask.data == False] = np.nan
    #                     #     gt_param_mean = np.nan_to_num(np.nanmean(gt_param_data, axis=(0, 1)))
    #
    #                     fig, ax = plt.subplots()
    #                     ax.set_title('{} {} {}'.format(dynamic_scatterer_name, parameter_name, ind), fontsize=16)
    #                     ax.plot(est_param_mean, common_grid.z, label='Estimated')
    #                     ax.plot(gt_param_mean, common_grid.z, label='True')
    #                     ax.legend()
    #                     ax.set_ylabel('Altitude [km]', fontsize=14)
    #                     self.tf_writer.add_figure(
    #                         tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name, ind),
    #                         figure=fig,
    #                         global_step=self.optimizer.iteration
    #                     )

    def scatter_plot_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for name, gt_cloud in self._ground_truth.items():
            est_scatterer = self.optimizer.medium.copy()
            if len(est_scatterer.shape)==3:
                gt_clouds = gt_cloud[None].copy()
                est_scatterers = est_scatterer[None]
            else:
                gt_clouds = gt_cloud.copy()
            colud_num = 0
            for gt_param, est_param in zip(gt_clouds, est_scatterers):
                mask = np.bitwise_or(gt_param>0,est_param>0)
                est_param = est_param[mask]
                gt_param = gt_param[mask]

                rho = np.corrcoef(est_param, gt_param)[1, 0]
                num_params = gt_param.size
                rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
                max_val = max(gt_param.max(), est_param.max())
                fig, ax = plt.subplots(1,1)
                ax.set_title(r'{} {}{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(name, 'extinction', colud_num, 100 * kwargs['percent'], rho),
                             fontsize=16)
                ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='b')
                ax.set_xlim([0, 1.1*max_val])
                ax.set_ylim([0, 1.1*max_val])
                ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
                ax.set_ylabel('Estimated', fontsize=14)
                ax.set_xlabel('True', fontsize=14)

                self.tf_writer.add_figure(
                    tag=kwargs['title'].format(name, 'extinction', colud_num),
                    figure=fig,
                    global_step=self.optimizer.iteration
                )
                colud_num +=1

    def write_image_list(self, global_step, images, titles, vmax=None):
        """
        Write an image list to tensorboardX.

        Parameters
        ----------
        global_step: integer,
            The global step of the optimizer.
        images: list
            List of images to be logged onto tensorboard.
        titles: list
            List of strings that will title the corresponding images on tensorboard.
        vmax: list or scalar, optional
            List or a single of scaling factor for the image contrast equalization
        """
        if np.isscalar(vmax) or vmax is None:
            vmax = [vmax] * len(images)

        assert len(images) == len(titles), 'len(images) != len(titles): {} != {}'.format(len(images), len(titles))
        assert len(vmax) == len(titles), 'len(vmax) != len(images): {} != {}'.format(len(vmax), len(titles))

        for image, title, vm in zip(images, titles, vmax):

            # for polarization
            if image.ndim == 4:
                stoke_title = ['V', 'U', 'Q', 'I']
                for v, stokes in zip(vm, image):
                    self.tf_writer.add_images(
                        tag=title + '/' + stoke_title.pop(),
                        img_tensor=(np.repeat(np.expand_dims(stokes, 2), 3, axis=2) / v),
                        dataformats='HWCN',
                        global_step=global_step
                    )

            # for polychromatic
            elif image.ndim == 3:
                self.tf_writer.add_images(
                    tag=title,
                    img_tensor=(np.repeat(np.expand_dims(image, 2), 3, axis=2) / vm),
                    dataformats='HWCN',
                    global_step=global_step
                )
            # for monochromatic
            else:
                self.tf_writer.add_image(
                    tag=title,
                    img_tensor=(image / vm),
                    dataformats='HW',
                    global_step=global_step
                )

    def monitor_images_scatter_plot(self, measurements, dilute_percent=0.1, ckpt_period=-1):
        """
        Monitor the Cross Validation process

        Parameters
        ----------
        cv_measurement: shdom.DynamicMeasurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        acquired_images = measurements.images
        num_images = len(acquired_images)

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Image_Scatter_Plot/view{}'.format(view) for view in range(num_images)],
            'acquired_images': acquired_images,
            'percent': dilute_percent,
        }
        self.add_callback_fn(self.images_scatter_plot_cbfn, kwargs)

    def images_scatter_plot_cbfn(self, kwargs):
        for estimated_image, acquired_images,title in zip(self.optimizer.images, kwargs['acquired_images'],kwargs['title']):
            estimated_image = copy.copy(estimated_image).ravel()
            acquired_images = copy.copy(acquired_images).ravel()
            rho = np.corrcoef(estimated_image, acquired_images)[1, 0]
            num_params = acquired_images.size
            rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
            max_val = max(acquired_images.max(), estimated_image.max())
            fig, ax = plt.subplots()
            ax.set_title(
                r'{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(0, 100 * kwargs['percent'], rho),
                fontsize=16)
            ax.scatter(acquired_images[rand_ind], estimated_image[rand_ind], facecolors='none', edgecolors='b')
            ax.set_xlim([0, 1.1 * max_val])
            ax.set_ylim([0, 1.1 * max_val])
            ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
            ax.set_ylabel('Estimated', fontsize=14)
            ax.set_xlabel('True', fontsize=14)

            self.tf_writer.add_figure(
                tag=title,
                figure=fig,
                global_step=self.optimizer.iteration
            )

    def monitor_cross_validation(self, cv_measurement, dilute_percent=0.1, ckpt_period=-1):
        """
        Monitor the Cross Validation process

        Parameters
        ----------
        cv_measurement: shdom.DynamicMeasurements
            The acquired images will be logged once onto tensorboard for comparison with the current state.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        acquired_images = cv_measurement.images
        sensor_type = cv_measurement.camera.sensor.type
        num_images = len(acquired_images)

        if sensor_type == 'RadianceSensor':
            vmax = [image.max() * 1.25 for image in acquired_images]
        elif sensor_type == 'StokesSensor':
            vmax = [image.reshape(image.shape[0], -1).max(axis=-1) * 1.25 for image in acquired_images]

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Cross Validation Retrieval/view{}'.format(view) for view in range(num_images)],
            'vmax': vmax
        }
        self.add_callback_fn(self.estimated_cross_validation_images_cbfn, kwargs)
        acq_titles = ['Cross Validation Acquired/view{}'.format(view) for view in range(num_images)]
        self.write_image_list(0, acquired_images, acq_titles, vmax=kwargs['vmax'])

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'Cross Validation/loss',
        }
        self.add_callback_fn(self.cv_loss_cbfn, kwargs)

        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'Cross Validation/image scatter plot',
            'percent': dilute_percent,
        }
        self.add_callback_fn(self.cv_image_scatter_plot_cbfn, kwargs)

    def cv_loss_cbfn(self, kwargs):
        """
        Callback function that is called (every optimizer iteration) for loss monitoring.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.tf_writer.add_scalar(kwargs['title'], self.optimizer.cv_loss, self.optimizer.iteration)

    def estimated_cross_validation_images_cbfn(self, kwargs):
        """
        Callback function the is called every optimizer iteration image monitoring is set.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        self.write_image_list(self.optimizer.iteration, self.optimizer.get_cross_validation_images(), kwargs['title'], kwargs['vmax'])

    def cv_image_scatter_plot_cbfn(self, kwargs):
        cv_image_estimated = np.array(self.optimizer.get_cross_validation_images()).ravel()
        cv_image = np.array(self.optimizer._cv_measurement.images).ravel()
        rho = np.corrcoef(cv_image_estimated, cv_image)[1, 0]
        num_params = cv_image.size
        rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
        max_val = max(cv_image.max(), cv_image_estimated.max())
        fig, ax = plt.subplots()
        ax.set_title(
            r'{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(0, 100 * kwargs['percent'], rho),
            fontsize=16)
        ax.scatter(cv_image[rand_ind], cv_image_estimated[rand_ind], facecolors='none', edgecolors='b')
        ax.set_xlim([0, 1.1 * max_val])
        ax.set_ylim([0, 1.1 * max_val])
        ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
        ax.set_ylabel('Estimated', fontsize=14)
        ax.set_xlabel('True', fontsize=14)

        self.tf_writer.add_figure(
            tag=kwargs['title'],
            figure=fig,
            global_step=self.optimizer.iteration
        )

    def monitor_cross_validation_scatterer_error(self, estimator_name, cv_ground_truth, ckpt_period=-1):
        """
        Monitor relative and overall mass error (epsilon, delta) as defined at:
          Amit Aides et al, "Multi sky-view 3D aerosol distribution recovery".

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': ['Cross Validation/delta {} at time {}', 'Cross Validation/epsilon {} at time {}']
        }
        self.add_callback_fn(self.cross_validation_scatterer_error_cbfn, kwargs)
        if hasattr(self, '_cv_ground_truth'):
            self._cv_ground_truth[estimator_name] = cv_ground_truth
        else:
            self._cv_ground_truth = OrderedDict({estimator_name: cv_ground_truth})

    def cross_validation_scatterer_error_cbfn(self, kwargs):
        """
        Callback function for monitoring parameter error measures.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for dynamic_scatterer_name, gt_temporary_scatterer in self._cv_ground_truth.items():
            est_scatterer = self.optimizer.get_cross_validation_medium(dynamic_scatterer_name).medium_list[0].estimators[dynamic_scatterer_name]
            for parameter_name, parameter in est_scatterer.estimators.items():
                gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                est_param = parameter.resample(gt_param.grid).data.flatten()
            # gt_param = gt_temporary_scatterer.scatterer.extinction
            # est_param = estimator_temporary_scatterer.scatterer.extinction.resample(gt_param.grid).data.flatten()
                gt_param = gt_param.data.flatten()

                delta = (np.linalg.norm(est_param, 1) - np.linalg.norm(gt_param, 1)) / np.linalg.norm(gt_param, 1)
                epsilon = np.linalg.norm((est_param - gt_param), 1) / np.linalg.norm(gt_param, 1)
                self.tf_writer.add_scalar(kwargs['title'][0].format(parameter_name, gt_temporary_scatterer.time), delta,
                                          self.optimizer.iteration)
                self.tf_writer.add_scalar(kwargs['title'][1].format(parameter_name, gt_temporary_scatterer.time), epsilon,
                                          self.optimizer.iteration)

    def monitor_cross_validation_scatter_plot(self, estimator_name, cv_ground_truth, dilute_percent=0.4, ckpt_period=-1, parameters='all'):
        """
        Monitor scatter plot of the parameters

        Parameters
        ----------
        estimator_name: str
            The name of the scatterer to monitor
        ground_truth: shdom.Scatterer
            The ground truth medium.
        dilute_precent: float [0,1]
            Precentage of (random) points that will be shown on the scatter plot.
        ckpt_period: float
           time [seconds] between updates. setting ckpt_period=-1 will log at every iteration.
        parameters: str,
           The parameters for which to monitor scatter plots. 'all' monitors all estimated parameters.
        """
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
            'title': 'Cross Validation {}/scatter_plot/{}{}',
            'percent': dilute_percent,
            'parameters': parameters
        }
        self.add_callback_fn(self.cross_validation_scatter_plot_cbfn, kwargs)
        if hasattr(self, '_ground_truth'):
            self._cv_ground_truth[estimator_name] = cv_ground_truth
        else:
            self._cv_ground_truth = OrderedDict({estimator_name: cv_ground_truth})

    def cross_validation_scatter_plot_cbfn(self, kwargs):
        """
        Callback function for monitoring scatter plot of parameters.

        Parameters
        ----------
        kwargs: dict,
            keyword arguments
        """
        for dynamic_scatterer_name, gt_temporary_scatterer in self._cv_ground_truth.items():
            est_scatterer = self.optimizer.get_cross_validation_medium(dynamic_scatterer_name).medium_list[0].estimators[dynamic_scatterer_name]
            for parameter_name, parameter in est_scatterer.estimators.items():

                    gt_param = getattr(gt_temporary_scatterer.scatterer, parameter_name)
                    est_param = parameter.resample(gt_param.grid)
                    est_param_data = copy.copy(est_param.data)
                    est_param_data = est_param_data[
                        (est_param.mask.resample(gt_param.grid).data == True)].ravel()

                    gt_param_data = copy.copy(gt_param.data)
                    gt_param_data = gt_param_data[
                        (est_param.mask.resample(gt_param.grid).data == True)].ravel()

                    rho = np.corrcoef(est_param_data, gt_param_data)[1, 0]
                    num_params = gt_param_data.size
                    rand_ind = np.unique(np.random.randint(0, num_params, int(kwargs['percent'] * num_params)))
                    max_val = max(gt_param_data.max(), est_param_data.max())
                    fig, ax = plt.subplots()
                    ax.set_title(r'{} {}{}: ${:1.0f}\%$ randomly sampled; $\rho={:1.2f}$'.format(dynamic_scatterer_name, parameter_name, 0, 100 * kwargs['percent'], rho),
                                 fontsize=16)
                    ax.scatter(gt_param_data[rand_ind], est_param_data[rand_ind], facecolors='none', edgecolors='b')
                    ax.set_xlim([0, 1.1*max_val])
                    ax.set_ylim([0, 1.1*max_val])
                    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
                    ax.set_ylabel('Estimated', fontsize=14)
                    ax.set_xlabel('True', fontsize=14)

                    self.tf_writer.add_figure(
                        tag=kwargs['title'].format(dynamic_scatterer_name, parameter_name, 0),
                        figure=fig,
                        global_step=self.optimizer.iteration
                    )

    def monitor_save3d(self, ckpt_period=-1):
        kwargs = {
            'ckpt_period': ckpt_period,
            'ckpt_time': time.time(),
        }
        self.add_callback_fn(self.save3d_cbfn, kwargs)

    def save3d_cbfn(self, kwargs):
        estimated_extinction_stack = []
        estimated_gridx_stack = []
        estimated_gridy_stack = []
        estimated_gridz_stack = []

        estimated_dynamic_medium = self.optimizer.medium.medium_list
        for medium_estimator in estimated_dynamic_medium:
            for estimator_name, estimator in medium_estimator.estimators.items():
                estimated_extinction = estimator.extinction
                estimated_extinction_stack.append(estimated_extinction.data)
                estimated_gridx_stack.append(estimated_extinction.grid.x)
                estimated_gridy_stack.append(estimated_extinction.grid.y)
                estimated_gridz_stack.append(estimated_extinction.grid.z)

        estimated_extinction_stack = np.stack(estimated_extinction_stack, axis=3)
        estimated_gridx_stack = np.stack(estimated_gridx_stack, axis=1)
        estimated_gridy_stack = np.stack(estimated_gridy_stack, axis=1)
        estimated_gridz_stack = np.stack(estimated_gridz_stack, axis=1)

        sio.savemat(os.path.join(self._dir, 'FINAL_3D_{}.mat'.format('extinction')),
                    {'estimated_extinction': estimated_extinction_stack, 'x': estimated_gridx_stack, 'y': estimated_gridy_stack, 'z': estimated_gridz_stack, 'iteration': self.optimizer.iteration})

    @property
    def callback_fns(self):
        return self._callback_fns

    @property
    def dir(self):
        return self._dir

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def tf_writer(self):
        return self._tf_writer


class OptimizationScript():
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
    def __init__(self,scatterer_name):
        #####################
        # Volume parameters #
        #####################
        # construct betas
        self.scatterer_name = scatterer_name
        self.data_dir = DEFAULT_DATA_ROOT + "CASS_50m_256x256x139_600CCN/lwcs_processed"
        self.files = glob.glob(join(self.data_dir, "*.npz"))
        # N_cloud = 110




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

    def optimization_args(self, parser):
        """
        Add common optimization arguments that may be shared across scripts.

        Parameters
        ----------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.

        Returns
        -------
        parser: argparse.ArgumentParser()
            parser initialized with basic arguments that are common to most rendering scripts.
        """
        parser.add_argument('--only_test',
                            action='store_true')
        parser.add_argument('--input_dir',
                            help='Path to an input directory where the forward modeling parameters are be saved. \
                                  This directory will be used to save the optimization results and progress.')
        parser.add_argument('--reload_path',
                            help='Reload an optimizer or checkpoint and continue optimizing from that point.')
        parser.add_argument('--log',
                            help='Write intermediate TensorBoardX results. \
                                  The provided string is added as a comment to the specific run.')
        parser.add_argument('--use_forward_grid',
                            action='store_true',
                            help='Use the same grid for the reconstruction. This is a sort of inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--use_forward_mask',
                            action='store_true',
                            help='Use the ground-truth cloud mask. This is an inverse crime which is \
                                  usefull for debugging/development.')
        parser.add_argument('--add_noise',
                            action='store_true',
                            help='currently only supports AirMSPI noise model. \
                                  See shdom.AirMSPINoise object for more info.')
        parser.add_argument('--n_jobs',
                            default=1,
                            type=int,
                            help='(default value: %(default)s) Number of jobs for parallel rendering. n_jobs=1 uses no parallelization')
        parser.add_argument('--globalopt',
                            action='store_true',
                            help='Global optimization with basin-hopping.'
                                 'For more info see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html')
        parser.add_argument('--maxiter',
                            default=1000,
                            type=int,
                            help='(default value: %(default)s) Maximum number of L-BFGS iterations.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--maxls',
                            default=30,
                            type=int,
                            help='(default value: %(default)s) Maximum number of line search steps (per iteration).'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--disp',
                            choices=[True, False],
                            default=True,
                            type=bool,
                            help='(default value: %(default)s) Display optimization progression.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--gtol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the maximum projected gradient.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--ftol',
                            default=1e-16,
                            type=np.float32,
                            help='(default value: %(default)s) Stop criteria for the relative change in loss function.'
                                 'For more info: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html')
        parser.add_argument('--stokes_weights',
                            nargs=4,
                            default=[1.0, 0.0, 0.0, 0.0],
                            type=float,
                            help='(default value: %(default)s) Loss function weights for stokes vector components [I, Q, U, V]')
        parser.add_argument('--loss_type',
                            choices=['l2', 'normcorr'],
                            default='l2',
                            help='Different loss functions for optimization. Currently only l2 is supported.')
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
        parser.add_argument("--beta1", type=float, default=0.01,
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


        self.args = parser.parse_args()
        self.args.save = 'logs/CF_CloudCT.{}'.format(time.strftime("%d-%b-%Y-%H:%M:%S"))

    def get_ground_truth(self):
        # Define the grid for reconstruction
        self.beta_max = 60
        beta_cloud = np.load(self.files[8])['lwc'] * self.beta_max
        print(self.files[8])
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
        grid = Grid(bbox, beta_cloud.shape)
        beta_air = 0.004
        w0_air = 0.912
        w0_cloud = 0.99
        volume = Volume(grid, beta_cloud, beta_air, w0_cloud, w0_air)
        volume.set_mask(beta_cloud>0.001)
        return volume



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
        ckpt_period = 120
        if self.args.log is not None:
            log_dir = os.path.join(self.args.save)
            writer = ArraySummaryWriter(log_dir)
            # writer.save_checkpoints(ckpt_period=-1)
            writer.monitor_loss(ckpt_period=ckpt_period)
            writer.monitor_images(measurements=measurements, ckpt_period=ckpt_period)

            # Compare estimator to ground-truth
            writer.monitor_scatterer_error(estimator_name=self.scatterer_name, ground_truth=ground_truth, ckpt_period=ckpt_period)
            writer.monitor_scatter_plot(estimator_name=self.scatterer_name, ground_truth=ground_truth, dilute_percent=0.4, parameters=[self.scatterer_name], ckpt_period=ckpt_period)
            # writer.monitor_horizontal_mean(estimator_name=self.scatterer_name, ground_truth=ground_truth, ground_truth_mask=ground_truth.get_mask(threshold=0.01))

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


    def get_measurements(self, volume):
        #######################
        # Cameras declaration #
        #######################
        bbox = volume.grid.bbox
        height_factor = 2

        focal_length = 1e-4  #####################
        sensor_size = np.array((3e-4, 3e-4)) / height_factor  #####################
        ps_max = 76

        pixels = np.array((ps_max, ps_max))

        N_cams = 9
        cameras = []
        volume_center = (bbox[:, 1] - bbox[:, 0]) / 2
        volume_center[-1] *= 1.8
        edge_z = bbox[-1,-1]
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
        N_render_gt = 1
        self.Np_gt = int(1e8)
        ########################
        # Atmosphere parameters#
        ########################
        sun_angles = np.array([180, 0]) * (np.pi / 180)
        g_cloud = 0.85
        rr_depth = 20
        rr_stop_prob = 0.05
        scene_rr = PytorchSceneSeed(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob,N_batches=N_render_gt, device=self.args.device)
        scene_rr.upscale_cameras(ps_max)
        scene_rr.set_cloud_mask(volume.cloud_mask)

        # scene_rr = PytorchSceneRR(volume, cameras, sun_angles, g_cloud, rr_depth, rr_stop_prob,N_batches=1, device=self.args.device)
        scene_rr.init_cuda_param(self.Np_gt, init=True)

        for i in range(N_render_gt):
            # cuda_paths = scene_rr.build_paths_list(self.Np_gt)
            scene_rr.build_paths_list(self.Np_gt)
            if i == 0:
                I_gt = scene_rr.render()#cuda_paths)
            else:
                I_gt += scene_rr.render()#cuda_paths)
        I_gt /= N_render_gt
        cuda_paths = None
        max_val = np.max(I_gt, axis=(1, 2))
        # visual.plot_images(I_gt, "GT")
        # plt.show()
        del (cuda_paths)


        return scene_rr, I_gt

    def get_NNoptimizer(self, ground_truth, measurements):
        """
        Define an Optimizer object

        Returns
        -------
        optimizer: shdom.Optimizer object
            An optimizer object.
        """



        # Initialize TensorboardX logger
        writer = self.get_summary_writer(measurements, ground_truth)

        optimizer = NNOptimizer(n_jobs=self.args.n_jobs)
        optimizer.set_measurements(measurements)
        optimizer.set_writer(writer)
        # Reload previous state
        if self.args.reload_path is not None:
            optimizer.load_state(self.args.reload_path)

        return optimizer


    def main(self):
        """
        Main optimization script
        """
        self.parse_arguments()
        self.args.cuda = (self.args.ngpu > 0) and torch.cuda.is_available()
        if self.args.cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.args.embed_fn, self.args.input_ch = get_embedder(self.args.multires, self.args.i_embed)
        self.args.skips = [self.args.netdepth / 2]
        self.args.device = device
        volume = self.get_ground_truth()
        scene_rr, I_gt = self.get_measurements(volume)
        local_optimizer = self.get_NNoptimizer(volume, I_gt)
        if self.args.only_test:
            self.test(local_optimizer, scene_rr, I_gt)

        else:
            self.train(local_optimizer,  scene_rr, I_gt)
        # Save optimizer state
        # save_dir =  self.args.input_dir
        # local_optimizer.save_state(os.path.join(save_dir, 'final_state.ckpt'))


    def train(self, optimizer,  scene_rr, measurements):
        torch_measurements = torch.tensor(measurements,device=self.args.device)
        model = Cloud_Flows(self.args).to(self.args.device)
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
        gridx,gridy,gridz = scene_rr.volume.grid.x_axis,scene_rr.volume.grid.y_axis,scene_rr.volume.grid.z_axis
        gridx, gridy, gridz = np.meshgrid(torch.tensor(gridx),
                                                torch.tensor(gridy),
                                                torch.tensor(gridz))
        grid = torch.tensor(np.array([gridx.ravel()[scene_rr.volume.cloud_mask.ravel()], gridy.ravel()[scene_rr.volume.cloud_mask.ravel()],
                                     gridz.ravel()[scene_rr.volume.cloud_mask.ravel()]]),dtype=torch.float).T.to(device=self.args.device)
        # grid *=1000
        # def forward_model(x):
        #     f = nn.AvgPool3d(2, 2)
        #     return f(x)



        error = nn.MSELoss()
        error = nn.Identity()

        if self.args.opt == 'sgd':
            NNoptimizer = optim.SGD(params=grad_vars, lr=self.args.lr,momentum=0.1)
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
        # if os.path.exists(self.args.save):
        #     shutil.rmtree(self.args.save)
        os.makedirs(self.args.save, exist_ok=True)
        # best_train_eps_convergence = []

        is_test=False
        print("===========Training===========")
        global is_rand
        is_rand = False
        # soft = torch.nn.Softplus(1)
        # est_volume = Volume(volume.grid, np.zeros_like(volume.beta_cloud), volume.beta_air, volume.w0_cloud, volume.w0_air)
        mc_renderer = MC_renderer().apply
        for epoch in range(self.args.nEpochs):
            model.rand(grid.shape[0])
            # averaged_variance = torch.mean(4.4444444444444444e-07 * torch_measurements)
            averaged_variance = 1.6675321e-10
            for iter in range(25):
                tic = time.time()
                NNoptimizer.zero_grad()
                x, loss_entropy = network_query_fn(grid, model, is_rand=is_rand, is_val=False,
                                                   is_test=is_test)  # alpha_mean.shape (B,N,1)

                # x = soft(x)
                # a,b = x.data.sort()
                print(x.max().item())
                print(x.min().item())
                x = (x+2) / 4 * self.beta_max #+ self.beta_max
                x = torch.squeeze(F.softplus(x))
                x = torch.clip(x,0,100)
                print(x.max().item())
                print(x.min().item())
                cloud = torch.zeros(scene_rr.volume.beta_cloud.shape,device=x.device)
                cloud[scene_rr.volume.cloud_mask] = x

                print(f'NN time {time.time()-tic}')
                tic = time.time()
                y = mc_renderer(cloud,torch_measurements,scene_rr,self.Np_gt)
                print(f'Renderer time {time.time()-tic}')
                loss_data = error(y)/(2*averaged_variance)
                loss = loss_data + loss_entropy * self.args.beta1
                # loss = error(torch.squeeze(x), torch.tensor(volume.lwc.data[optimizer.mask.data],dtype=torch.float,device=device))
                # loss_t += loss
                # output_np = xx.detach().cpu().numpy()

                # Calculating gradients
                try:
                    loss.backward()
                except:
                    print()
                # for p in model.parameters():
                #     print(p.grad)
                NN_loss_list.append(loss_data.mean().item())
                optimizer._loss = [loss_data.mean().item(),loss_entropy.mean().item()]
                optimizer.set_medium_estimator(cloud.detach().cpu().numpy())
                optimizer._images = scene_rr.I_opt
                # optimizer._medium.set_state(np.squeeze(cloud.detach().cpu().numpy()))
                optimizer.callback(None)
                relative_error_list.append(optimizer.writer.epsilon)

                print('Epoch {}, Iter {}: Image Loss: {} Entropy Loss: {} relative_error: {}'.format(epoch,iter, loss_data,
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

    def test(self, optimizer,  scene_rr, measurements):
        torch_measurements = torch.tensor(measurements,device=self.args.device)
        model = Cloud_Flows(self.args).to(self.args.device)
        # model = nn.DataParallel(model).to(device)
        # if not self.args.no_reload:
        if self.args.ft_path is None:
            NotImplementedError()
        ckpt_path = os.path.join(self.args.ft_path)
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        pretrained_dict = ckpt['network_fn_state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        network_query_fn = lambda inputs, network_fn, is_rand,is_val,  is_test: self.run_network(inputs,
                                                                                             network_fn, is_rand,is_val,
                                                                                             is_test,
                                                                                             embed_fn=self.args.embed_fn,
                                                                                             netchunk=self.args.netchunk_per_gpu * self.args.ngpu)
        # f_x = ForwardModel(self.args.input_dir)
        # x_gt = ground_truth.lwc.data
        gridx,gridy,gridz = scene_rr.volume.grid.x_axis,scene_rr.volume.grid.y_axis,scene_rr.volume.grid.z_axis
        gridx, gridy, gridz = np.meshgrid(torch.tensor(gridx),
                                                torch.tensor(gridy),
                                                torch.tensor(gridz))
        grid = torch.tensor(np.array([gridx.ravel()[scene_rr.volume.cloud_mask.ravel()], gridy.ravel()[scene_rr.volume.cloud_mask.ravel()],
                                     gridz.ravel()[scene_rr.volume.cloud_mask.ravel()]]),dtype=torch.float).T.to(device=self.args.device)
        # grid *=1000
        # def forward_model(x):
        #     f = nn.AvgPool3d(2, 2)
        #     return f(x)




        # gd_relative_error_convergence = []
        relative_error_list = []
        NN_loss_list = []
        # if os.path.exists(self.args.save):
        #     shutil.rmtree(self.args.save)
        # os.makedirs(self.args.save, exist_ok=True)
        # best_train_eps_convergence = []

        is_test=False
        print("===========Testing===========")
        global is_rand
        is_rand = False
        soft = torch.nn.Softplus(1)
        # est_volume = Volume(volume.grid, np.zeros_like(volume.beta_cloud), volume.beta_air, volume.w0_cloud, volume.w0_air)
        mc_renderer = MC_renderer().apply

        error = nn.Identity()
        cloud_list = []
        images = []
        for iter in range(200):
            model.rand(grid.shape[0])
            x, loss_entropy = network_query_fn(grid, model, is_rand=is_rand, is_val=False,
                                                   is_test=is_test)  # alpha_mean.shape (B,N,1)

            x = x * 60 + 60
            x = torch.squeeze(F.softplus(x))
            x = torch.clip(x, 0, 300)

            cloud = torch.zeros(scene_rr.volume.beta_cloud.shape,device=x.device)
            cloud[scene_rr.volume.cloud_mask] = x
            cloud_list.append(cloud.detach().cpu().numpy())


            y = mc_renderer(cloud,torch_measurements,scene_rr,int(self.Np_gt))

            loss = error(y)
            # loss = error(torch.squeeze(x), torch.tensor(volume.lwc.data[optimizer.mask.data],dtype=torch.float,device=device))
            # loss_t += loss
            # output_np = xx.detach().cpu().numpy()

            # Calculating gradients
            # for p in model.parameters():
            #     print(p.grad)
            images.append(scene_rr.I_opt.copy())
            NN_loss_list.append(loss.mean().item())
            optimizer._loss = loss.mean().item()
            optimizer.set_medium_estimator(cloud.detach().cpu().numpy())
            optimizer._images = scene_rr.I_opt
            # optimizer._medium.set_state(np.squeeze(cloud.detach().cpu().numpy()))
            optimizer.callback(None)
            relative_error_list.append(optimizer.writer.epsilon)

            print('Cloud #{}: Image Loss: {} Entropy Loss: {} relative_error: {}'.format(iter, loss,
                                                                                        loss_entropy.item(),
                                                                                        relative_error_list[
                                                                                            -1]))

        print('Total image Loss: {}+-{} relative_error: {}+-{}'.format(np.mean(NN_loss_list),np.std(NN_loss_list),
                                                            np.mean(relative_error_list),np.std(relative_error_list)))
        path = os.path.join(optimizer.writer.dir,
                            'results.np')
        np.save(path,{'est_clouds':cloud_list,'gt_clouds':optimizer.writer._ground_truth['cloud'],'loss':NN_loss_list,'images':images})
        gt = optimizer.writer._ground_truth['cloud']
        e = [(c-gt)[gt>0] for c in cloud_list]
        e = np.array(e)
        voxel_std = np.std(e,0)
        voxel_mean = np.mean(e, 0)
        plt.hist(voxel_std, 30)
        plt.show()
        plt.hist(voxel_mean, 30)
        plt.show()

        # fig2, ax2 = plt.subplots(nrows=1, ncols=1)  # two axes on figure
        # ax2.set_aspect('equal', adjustable='box')
        colors = np.array(
            ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan",
             "magenta"])

        gt_flat = gt[gt > 0]
        i = np.random.permutation(gt_flat.size)[:len(colors)]
        for c in cloud_list:
            c = c[gt>0]
            plt.scatter(gt_flat[i], c[i],c=colors)
        plt.plot([0, 100], [0, 100], c='r', ls='--')
        plt.xlabel('GT')
        plt.ylabel('Estimated')
        plt.show()

        i = np.random.permutation(gt_flat.size)[:len(colors)]
        e = [c[gt>0] for c in cloud_list]
        e = np.array(e)
        voxel_std = np.std(e,0)
        voxel_mean = np.mean(e, 0)
        plt.errorbar(gt_flat[i], voxel_mean[i], yerr=voxel_std[i], fmt="o", ecolor=colors)
        plt.scatter(gt_flat[i], voxel_mean[i], c=colors)

        plt.plot([0, 100], [0, 100], c='r', ls='--')
        plt.xlabel('GT')
        plt.ylabel('Estimated')
        plt.show()

        i = np.random.permutation(gt_flat.size)[:50]

        e = [c[gt > 0] for c in cloud_list]
        e = np.array(e)
        voxel_std = np.std(e, 0)
        voxel_mean = np.mean(e, 0)
        plt.errorbar(gt_flat[i], voxel_mean[i], yerr=voxel_std[i], fmt="o")
        plt.scatter(gt_flat[i], voxel_mean[i])

        plt.plot([0, 100], [0, 100], c='r', ls='--')
        plt.xlabel('GT')
        plt.ylabel('Estimated')
        plt.show()

        plt.plot(gt[gt>0],voxel_std,'o')
        plt.show()



class MC_renderer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, measurements, scene_rr,np):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        print("RESAMPLING PATHS ")
        scene_rr.volume.beta_cloud = torch.squeeze(input).detach().cpu().numpy().astype(
            float_reg)
        scene_rr.init_cuda_param(np)
        # cuda_paths = scene_rr.build_paths_list(np)
        scene_rr.build_paths_list(np)
        # I_opt, total_grad = scene_rr.render(cuda_paths, I_gt=measurements.detach().cpu().numpy(), to_torch=True)
        I_opt, total_grad = scene_rr.render(I_gt=measurements.detach().cpu().numpy(), to_torch=True)
        total_grad *= (measurements.numel())
        I_opt = torch.tensor(I_opt, dtype=input.dtype,device=total_grad.device) #* (measurements.numel())
        error = nn.MSELoss()
        scene_rr.I_opt = I_opt.detach().cpu().numpy()
        print('Done image consistency loss stage')
        total_grad[torch.logical_not(torch.isfinite(total_grad))] = 0
        ctx.save_for_backward(torch.squeeze(total_grad))
        loss = error(I_opt,measurements) * (measurements.numel())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # import pdb
        # pdb.set_trace()
        gradient, = ctx.saved_tensors
        # print(gradient.shape)
        return gradient * grad_output.clone(), None, None, None


if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()



