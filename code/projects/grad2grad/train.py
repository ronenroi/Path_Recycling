import torch
import torch.nn as nn

from cuda_utils import *
from projects.grad2grad.noise2noise import Noise2Noise
from projects.grad2grad.datasets import load_dataset
from argparse import ArgumentParser
cuda.select_device(0)

def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('-t', '--train-dir', help='training set path', default='/home/roironen/Path_Recycling/code/generated_data_noise2clean/train')
    parser.add_argument('-v', '--valid-dir', help='test set path', default='/home/roironen/Path_Recycling/code/generated_data_noise2clean/val')
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='/home/roironen/Path_Recycling/code/projects/grad2grad/output')
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('--report-interval', help='batch report interval', default=5, type=int)
    parser.add_argument('-ts', '--train-size', help='size of train dataset', default=-1, type=int)
    parser.add_argument('-vs', '--valid-size', help='size of valid dataset', default=-1, type=int)
    parser.add_argument('-min_iter',  help='minimum gradint itertaion step', default=0, type=int)
    parser.add_argument('-max_iter', help='max gradint itertaion step', default=100, type=int)
    parser.add_argument('-net', help='network type', default='UNET', type=str)


    # Training hyperparameters
    parser.add_argument('-lr', '--learning-rate', help='learning rate', default=0.00001, type=float)
    parser.add_argument('-a', '--adam', help='adam parameters', nargs='+', default=[0.9, 0.99, 1e-8], type=list)
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=10, type=int)
    parser.add_argument('-w', '--num-workers', help='num workers', default=4, type=int)

    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=100, type=int)
    parser.add_argument('-l', '--loss', help='loss function', choices=['l1', 'l2', 'hdr'], default='l2', type=str)
    parser.add_argument('--cuda', help='use cuda', action='store_false')
    parser.add_argument('--plot-stats', help='plot stats after every epoch', action='store_false')

    # Corruption parameters
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['mc'], default='mc', type=str)
    # parser.add_argument('-p', '--noise-param', help='noise parameter (e.g. std for gaussian)', default=50, type=float)
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    # parser.add_argument('-c', '--crop-size', help='random crop size', default=0, type=int)
    # parser.add_argument('--image-size', help='image size', default=512, type=int)

    parser.add_argument('--clean-targets', help='use clean targets for training', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    """Trains Noise2Noise."""

    # Parse training parameters
    params = parse_args()

    # Train/valid datasets
    train_loader = load_dataset(params.train_dir, params.train_size, params, shuffled=True)
    valid_loader = load_dataset(params.valid_dir, params.valid_size, params, shuffled=False)

    # Initialize model and train
    n2n = Noise2Noise(params, trainable=True)
    n2n.train(train_loader, valid_loader)