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
            nn.Linear(size,int(size/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(size / 2), int(size / 2))
        )
        self.linear2 = nn.Linear(size,int(size/2))
        self.relu1 = nn.ReLU(inplace=True)

        self.block2 = nn.Sequential(
            nn.Linear(int(size / 2),size),
            nn.ReLU(inplace=True),
        )


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
        x = x + b1

        # Final activation
        return x