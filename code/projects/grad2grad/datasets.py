#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader


import os, glob
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset

    # Instantiate appropriate dataset class
    dataset = MonteCarloDataset(root_dir, redux, params.crop_size, params.image_size)


    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, image_size=256):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.image_size = image_size

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs

    def _pad(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        c, w, h = img_list[0].shape
        assert w <= self.image_size and h <= self.image_size, \
            f'Error: Pad size: {self.image_size}, Image size: ({w}, {h})'
        padded_imgs = []


        for img in img_list:
            padded_imgs.append(tvF.pad(img, (int((self.image_size-w)/2),int((self.image_size-h)/2))))


        return padded_imgs

    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""

        return len(self.imgs)




class MonteCarloDataset(AbstractDataset):
    """Class for dealing with Monte Carlo rendered images."""

    def __init__(self, root_dir, redux, crop_size, image_size):
        """Initializes Monte Carlo image dataset."""

        super(MonteCarloDataset, self).__init__(root_dir, redux, crop_size, image_size)

        # Rendered images directories
        self.root_dir = root_dir
        self.imgs = [f for f in glob.glob(os.path.join(root_dir, 'source*.pt'))]
        self.params = [f for f in glob.glob(os.path.join(root_dir, '*params*'))]

        if redux:
            self.imgs = self.imgs[:redux]


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Use converged image, if requested

        target_fname = self.imgs[index].replace('source', 'target')
        file_ext = '.pt'
        target_fname = os.path.splitext(target_fname)[0] + file_ext
        target_path = os.path.join(self.root_dir, 'target', target_fname)
        target = torch.load(target_path).permute(2,0,1)

        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        source = torch.load(render_path).permute(2,0,1)
        buffers = [source, target]

        # Crop
        if self.crop_size != 0:
            buffers = [b for b in self._random_crop(buffers)]
        else:
            buffers = [b for b in self._pad(buffers)]

        # Stack buffers to create input volume
        source = buffers[0]
        target = buffers[1]

        return source, target

    # def __getitem__(self, index):
    #     """Retrieves image from folder and corrupts it."""
    #
    #     # Use converged image, if requested
    #
    #     target_fname = self.imgs[index].replace('source', 'target')
    #     file_ext = '.png'
    #     target_fname = os.path.splitext(target_fname)[0] + file_ext
    #     target_path = os.path.join(self.root_dir, 'target', target_fname)
    #     target = Image.open(target_path).convert('RGB')
    #
    #     render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
    #     source = Image.open(render_path).convert('RGB')
    #     buffers = [source, target]
    #
    #     # Crop
    #     if self.crop_size != 0:
    #         buffers = [tvF.to_tensor(b) for b in self._random_crop(buffers)]
    #     else:
    #         buffers = [tvF.to_tensor(b) for b in self._pad(buffers)]
    #
    #     # Stack buffers to create input volume
    #     source = buffers[0]
    #     target = buffers[1]
    #
    #     return source, target