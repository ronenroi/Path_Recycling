#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F

from projects.grad2grad.unet import *
from projects.grad2grad.linear import Linear, LinearWeights
from projects.grad2grad.grad_utils import *
from torch.utils.tensorboard import SummaryWriter

import os
import json


class Noise2Noise(object):
    """Implementation of Noise2Noise from Lehtinen et al. (2018)."""

    def __init__(self, params, trainable):
        """Initializes model."""

        self.p = params
        self.net = params.net
        self.trainable = trainable
        self._compile()
        # Create directory for model checkpoints, if none existent
        ckpt_dir_name = f'{datetime.now():{self.p.noise_type}-%H%M}'

        if trainable:
            if self.p.ckpt_overwrite:
                ckpt_dir_name = self.p.noise_type

            self.ckpt_dir = os.path.join(self.p.ckpt_save_path, ckpt_dir_name)

            if not os.path.isdir(self.p.ckpt_save_path):
                os.mkdir(self.p.ckpt_save_path)
            if not os.path.isdir(self.ckpt_dir):
                os.mkdir(self.ckpt_dir)
            self._writer =  SummaryWriter(self.ckpt_dir)



    def _compile(self):
        """Compiles model (architecture, loss function, optimizers, etc.)."""

        print('Noise2Noise: Learning Image Restoration without Clean Data (Lethinen et al., 2018)')

        # Model
        if self.net == 'UNET':
            self.model = UNet(in_channels=11)
        elif self.net == 'UNET2':
            self.model = UNet2(in_channels=11)
        elif self.net == 'Linear':
            self.model = Linear(size=32*10)#.double()
        elif self.net == 'LinearWeights':
            self.model = LinearWeights(size=32 * 10)

        # Set optimizer and loss, if in training mode
        if self.trainable:
            self.optim = Adam(self.model.parameters(),
                              lr=self.p.learning_rate,
                              betas=self.p.adam[:2],
                              eps=self.p.adam[2])

            # Learning rate adjustment
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim,
                patience=self.p.nb_epochs/4, factor=0.5, verbose=True)

            # Loss function
            if self.p.loss == 'hdr':
                self.loss = HDRLoss()
            elif self.p.loss == 'l2':
                self.loss = nn.MSELoss()
            elif self.p.loss == 'cos':
                self.loss = CosineLoss()

            else:
                self.loss = nn.L1Loss()

        # CUDA support
        self.use_cuda = torch.cuda.is_available() and self.p.cuda
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                self.loss = self.loss.cuda()


    def _print_params(self):
        """Formats parameters to print when training."""

        print('Training parameters: ')
        self.p.cuda = self.use_cuda
        param_dict = vars(self.p)
        pretty = lambda x: x.replace('_', ' ').capitalize()
        print('\n'.join('  {} = {}'.format(pretty(k), str(v)) for k, v in param_dict.items()))
        print()


    def save_model(self, epoch, stats, first=False):
        """Saves model to files; can be overwritten at every epoch to save disk space."""



        # Save checkpoint dictionary
        if self.p.ckpt_overwrite:
            fname_unet = '{}/n2n-{}.pt'.format(self.ckpt_dir, self.p.noise_type)
        else:
            valid_loss = stats['valid_loss'][epoch]
            fname_unet = '{}/n2n-epoch{}-{:>1.5f}.pt'.format(self.ckpt_dir, epoch + 1, valid_loss)
        print('Saving checkpoint to: {}\n'.format(fname_unet))
        torch.save(self.model.state_dict(), fname_unet)

        # Save stats to JSON
        fname_dict = '{}/n2n-stats.json'.format(self.ckpt_dir)
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)


    def load_model(self, ckpt_fname):
        """Loads model from checkpoint file."""

        print('Loading checkpoint from: {}'.format(ckpt_fname))
        if self.use_cuda:
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))

    def _pad(self, img, size=512):
        c, w, h = img.shape
        assert w <= size and h <= size, \
            f'Error: Pad size: {size}, Image size: ({w}, {h})'


        return tvF.pad(img, (int((size-w)/2),int((size-h)/2)))

    def _unpad(self, img, pad=6):

        return img[:,pad:-pad,pad:-pad]


    def _on_epoch_end(self, stats, train_loss, epoch, epoch_start, valid_loader):
        """Tracks and saves starts after each epoch."""

        # Evaluate model on validation set
        print('\rTesting model on validation set... ', end='')
        epoch_time = time_elapsed_since(epoch_start)[0]
        valid_loss, valid_time, valid_psnr, naive_psnr = self.eval(valid_loader)
        show_on_epoch_end(epoch_time, valid_time, valid_loss, valid_psnr)

        # Decrease learning rate if plateau
        self.scheduler.step(valid_loss)

        # Save checkpoint
        stats['train_loss'].append(train_loss)
        stats['valid_loss'].append(valid_loss)
        stats['valid_psnr'].append(valid_psnr)
        stats['naive_psnr'].append(naive_psnr)
        if epoch % 10==0:
            self.save_model(epoch, stats, epoch == 0)

        # Plot stats
        if self.p.plot_stats:
            loss_str = f'{self.p.loss.upper()} loss'
            plot_per_epoch(self.ckpt_dir, loss_str, [stats['train_loss'], stats['valid_loss']], ['train_loss', 'valid_loss'],yaxis='log')
            plot_per_epoch(self.ckpt_dir, 'Valid PSNR (dB)', [stats['valid_psnr'], stats['naive_psnr']], ['Denoised', 'Naive Average'])
            self._writer.add_scalars('Loss', {'Train loss': train_loss,'Validation loss': valid_loss}, epoch)
            self._writer.add_scalar('Valid PSNR (dB)', valid_psnr, epoch)
            #self._writer.add_scalar('lr', self.scheduler._last_lr, epoch)

    def test(self, image):
        """Evaluates denoiser on test image."""

        self.model.train(False)
        image = image.permute(2,0,1)
        image = self._pad(image)
        image = self.model(image[None,...])[0]
        image = self._unpad(image)
        return image.permute(1,2,0)

        # Create montage and save images
        #print('Saving images and montages to: {}'.format(save_path))
        #for i in range(len(source_imgs)):
        #    img_name = test_loader.dataset.imgs[i]
        #    create_montage(img_name, self.p.noise_type, save_path, source_imgs[i], denoised_imgs[i], clean_imgs[i], show)


    def eval(self, valid_loader):
        """Evaluates denoiser on validation set."""

        self.model.train(False)

        valid_start = datetime.now()
        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()
        naive_solution_meter = AvgMeter()


        for batch_idx, (source, target) in enumerate(valid_loader):
            if self.use_cuda:
                source = source.cuda()
                target = target.cuda()

            # Denoise
            source_denoised = self.model(source)
            averaged_source = torch.mean(source[:,1:,...],1)
            # Update loss
            loss = self.loss(source_denoised, target)
            loss_meter.update(loss.item())

            # Compute PSRN
            #source_denoised = reinhard_tonemap(source_denoised)
            # source_denoised[source_denoised < 0] = 0
            # TODO: Find a way to offload to GPU, and deal with uneven batch sizes
            source_denoised = source_denoised.cpu()
            target = target.cpu()
            averaged_source = averaged_source.cpu()
            for i in range(source_denoised.shape[0]):
                psnr_meter.update(psnr(torch.squeeze(source_denoised[i]), torch.squeeze(target[i])).item())
                naive_solution_meter.update(psnr(torch.squeeze(averaged_source[i]), torch.squeeze(target[i])).item())
        valid_loss = loss_meter.avg
        valid_time = time_elapsed_since(valid_start)[0]
        psnr_avg = psnr_meter.avg
        psnr_naive = naive_solution_meter.avg
        return valid_loss, valid_time, psnr_avg, psnr_naive


    def train(self, train_loader, valid_loader):
        """Trains denoiser on training set."""

        self.model.train(True)

        self._print_params()
        num_batches = len(train_loader)
        assert num_batches % self.p.report_interval == 0, 'Report interval must divide total number of batches'

        # Dictionaries of tracked stats
        stats = {'noise_type': self.p.noise_type,
                 # 'noise_param': self.p.noise_param,
                 'train_loss': [],
                 'valid_loss': [],
                 'valid_psnr': [],
                 'naive_psnr': []}

        # Main training loop
        train_start = datetime.now()
        for epoch in range(self.p.nb_epochs):
            print('EPOCH {:d} / {:d}'.format(epoch + 1, self.p.nb_epochs))

            # Some stats trackers
            epoch_start = datetime.now()
            train_loss_meter = AvgMeter()
            loss_meter = AvgMeter()
            time_meter = AvgMeter()

            # Minibatch SGD
            for batch_idx, (source, target) in enumerate(train_loader):
                batch_start = datetime.now()
                progress_bar(batch_idx, num_batches, self.p.report_interval, loss_meter.val)

                if self.use_cuda:
                    source = source.cuda()
                    target = target.cuda()

                # Denoise image
                source_denoised = self.model(source)

                loss = self.loss(source_denoised, target)
                loss_meter.update(loss.item())

                # Zero gradients, perform a backward pass, and update the weights
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # Report/update statistics
                time_meter.update(time_elapsed_since(batch_start)[1])
                # if (batch_idx + 1) % self.p.report_interval == 0 and batch_idx:
                show_on_report(batch_idx, num_batches, loss_meter.avg, time_meter.avg)
                train_loss_meter.update(loss_meter.avg)
                loss_meter.reset()
                time_meter.reset()

            # Epoch end, save and reset tracker
            self._on_epoch_end(stats, train_loss_meter.avg, epoch, epoch_start, valid_loader)
            train_loss_meter.reset()

        train_elapsed = time_elapsed_since(train_start)[0]
        print('Training done! Total elapsed time: {}\n'.format(train_elapsed))


class HDRLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=0.01):
        """Initializes loss with numerical stability epsilon."""

        super(HDRLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        """Computes loss by unpacking render buffer."""

        loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return torch.mean(loss.view(-1))



class CosineLoss(nn.Module):
    """High dynamic range loss."""

    def __init__(self, eps=1e-10):
        """Initializes loss with numerical stability epsilon."""

        super(CosineLoss, self).__init__()
        self._eps = eps


    def forward(self, denoised, target):
        # maximize average cosine similarity
        batch_size = denoised.shape[0]
        loss = -F.cosine_similarity(denoised.reshape(batch_size,-1), target.reshape(batch_size,-1), eps=self._eps).mean()
        # loss = ((denoised - target) ** 2) / (denoised + self._eps) ** 2
        return loss
# def loss_func(feat1, feat2):
#     # minimize average magnitude of cosine similarity
#     return F.cosine_similarity(feat1, feat2).abs().mean()
#
# def loss_func(feat1, feat2):
#     # minimize average cosine similarity
#     return F.cosine_similarity(feat1, feat2).mean()
#
# def loss_func(feat1, feat2):
#     # maximize average magnitude of cosine similarity
#     return -F.cosine_similarity(feat1, feat2).abs().mean()
#
# def loss_func(feat1, feat2):
#     # maximize average cosine similarity
#     return -F.cosine_similarity(feat1, feat2).mean()