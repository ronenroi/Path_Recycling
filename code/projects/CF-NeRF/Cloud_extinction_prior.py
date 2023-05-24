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

import numpy as np

my_lib_path = os.path.abspath('./')
sys.path.append(my_lib_path)
from os.path import join
import glob
from scipy.io import savemat


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
        self.data_dir = "/home/roironen/pytorch3d/projects/CT/data/CASS_50m_256x256x139_600CCN/lwcs_processed"
        self.files = glob.glob(join(self.data_dir, "*.npz"))
        # N_cloud = 110

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
        # parser = self.optimization_args(parser)
        # parser = self.medium_args(parser)
        # parser = self.neuralnetwork_args(parser)


        self.args = parser.parse_args()
        self.args.save = 'logs/CF_CloudCT.{}'.format(time.strftime("%d-%b-%Y-%H:%M:%S"))

    def get_ground_truth(self):
        import torch.fft
        # Define the grid for reconstruction
        power3 = []
        power2 = []
        for file in self.files[:]:
            lwc = np.load(file)['lwc']
            lwc = torch.tensor(lwc.astype(float),device='cuda')
            # FS3 = np.fft.fftshift(np.abs(np.fft.fftn(lwc)**2))
            FS3 = torch.abs(torch.fft.fftn(lwc,dim=[0,1,2]) ) ** 2
            power3.append(FS3.cpu().numpy())
            # FS2 = np.fft.fftshift(np.abs(np.fft.fft2(lwc,axes=(0,1))**2),axes=(0,1))
            # power2.append(FS2)
        savemat('lwc_PS.mat',{'power3':np.mean(power3,0)})
        # savemat('lwc_PS.mat',{'power3':power3,'power2':power2})


    def main(self):
        """
        Main optimization script
        """
        self.parse_arguments()
        self.get_ground_truth()






if __name__ == "__main__":
    script = OptimizationScript(scatterer_name='cloud')
    script.main()



