#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, size):
        """Initializes U-Net."""

        super(Linear, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(size,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        self.linear2 = nn.Linear(size,512)
        self.relu1 = nn.ReLU(inplace=True)

        self.block2 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
        )
        self.linear3 = nn.Linear(size, 32)

        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        b1 = self.block1(x)
        x1 = self.linear2(x)
        b1 = self.relu1(b1 + x1)
        b1 = self.block2(b1)
        b1 = self.linear3(x) + b1

        # Final activation
        return b1


class LinearWeights(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, size):
        """Initializes U-Net."""

        super(LinearWeights, self).__init__()
        out_size = 10
        self.block1 = nn.Sequential(
            nn.Linear(size,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        self.linear2 = nn.Linear(size,512)
        self.relu1 = nn.ReLU(inplace=True)

        self.block2 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )
        self.linear3 = nn.Linear(size, 10)
        self.softmax = nn.Softmax(dim=1)
        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        b1 = self.block1(x)
        x1 = self.linear2(x)
        b1 = self.relu1(b1 + x1)
        b1 = self.block2(b1)
        b1 = self.linear3(x) + b1
        # b1 = b1 / torch.sum(b1, dim=1)[...,None]
        x = x.reshape(x.shape[0],10,32)
        # b1 = torch.mean(b1.reshape(x.shape[0],10,32),dim=1)

        b1 = torch.mean(b1[...,None] * x,dim=1)
        return b1

class LinearWeights1(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, size):
        """Initializes U-Net."""

        super(LinearWeights1, self).__init__()
        out_size = 10
        self.block1 = nn.Sequential(
            nn.Linear(size,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        self.linear2 = nn.Linear(size,512)
        self.relu1 = nn.ReLU(inplace=True)

        self.block2 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, size),
        )
        self.linear3 = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=1)
        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """
        b1 = self.block1(x)
        x1 = self.linear2(x)
        b1 = self.relu1(b1 + x1)
        b1 = self.block2(b1)
        b1 = self.linear3(x) + b1
        # b1 = b1 / torch.sum(b1, dim=1)[...,None]
        # x = x.reshape(x.shape[0],10,32)
        b1 = torch.mean(b1.reshape(x.shape[0],10,32),dim=1)

        # b1 = torch.mean(b1[...,None] * x,dim=1)
        # Final activation
        return b1