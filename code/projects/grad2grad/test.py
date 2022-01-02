


from projects.dev.grad2grad.noise2noise import Noise2Noise
from projects.dev.grad2grad.utils import plot_per_epoch
from projects.dev.grad2grad.datasets import load_dataset
from argparse import ArgumentParser

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, Vector3f
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import  write_bitmap
from projects.dev.grad2grad.my_render import render_torch, G2G


import torch
import time

def parse_args():
    """Command-line argument parser for testing."""

    # New parser
    parser = ArgumentParser(description='PyTorch implementation of Noise2Noise from Lehtinen et al. (2018)')

    # Data parameters
    parser.add_argument('--ckpt-save-path', help='checkpoint save path', default='/home/roironen/mitsuba2/projects/dev/grad2grad/data/cbox/')
    parser.add_argument('--cuda', help='use cuda', action='store_false')
    parser.add_argument('-s', '--seed', help='fix random seed', type=int)
    parser.add_argument('-c', '--crop-size', help='random crop size', default=0, type=int)
    parser.add_argument('--image-size', help='image size', default=512, type=int)
    parser.add_argument('--ckpt-overwrite', help='overwrite model checkpoint on save', action='store_true')
    parser.add_argument('-n', '--noise-type', help='noise type',
        choices=['mc'], default='mc', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    """Test Noise2Noise."""

    # Parse training parameters
    parse_params = parse_args()
    Thread.thread().file_resolver().append('cbox')
    scene = load_file('/home/roironen/mitsuba2/resources/data/docs/scenes/variants_cbox_rgb.xml')
    # Find differentiable scene parameters
    params = traverse(scene)
    # Discard all parameters except for one we want to differentiate
    params.keep(['red.reflectance.value', 'green.reflectance.value', 'white.reflectance.value'])
    #params.keep(        ['red.reflectance.value', 'green.reflectance.value'])
    # Print the current value and keep a backup copy
    param_ref = []
    for k, _ in params.properties.items():
        param_ref.append(params[k].torch())
    print(param_ref)
    # Render a reference image (no derivatives used yet)
    image_ref = render_torch(scene, spp=100)
    crop_size = scene.sensors()[0].film().crop_size()
    write_bitmap('/home/roironen/mitsuba2/projects/dev/grad2grad/test_output/out_ref.png', image_ref, crop_size)


    # Construct a PyTorch Adam optimizer that will adjust 'params_torch'
    objective = torch.nn.MSELoss()

    # Initialize trained model
    n2n = Noise2Noise(parse_params, trainable=False)
    n2n.load_model('/home/roironen/mitsuba2/projects/dev/grad2grad/data/cbox/mc-1407/n2n-epoch24-0.00050.pt')
    time_a = time.time()
    spp = 5
    iterations = 100
    err = []
    loss = []
    for use_nn in [False, True]:
        # Change all parameters initial value
        for k, _ in params.properties.items():
            params[k] = [.9, .9, .9]
        params.update()

        # Which parameters should be exposed to the PyTorch optimizer?
        params_torch = params.torch()
        opt = torch.optim.Adam(params_torch.values(), lr=0.05)

        for it in range(iterations):
            # Zero out gradients before each iteration
            opt.zero_grad()

            # Perform a differentiable rendering of the scene
            image = render_torch(scene, params=params, unbiased=True,
                                 spp=spp,**params_torch)
            write_bitmap('/home/roironen/mitsuba2/projects/dev/grad2grad/test_output/noisy_out_%03i.png' % it, image, crop_size)
            if use_nn:
                image = n2n.test(image)
            # image[:150, :, :] = 0
            # image[151:, :, :] = 0
            # image_ref1 = image_ref
            # image_ref1[:150, :, :] = 0
            # image_ref1[151:, :, :] = 0
            write_bitmap('/home/roironen/mitsuba2/projects/dev/grad2grad/test_output/clean_out_%03i.png' % it, image, crop_size)
            # Objective: MSE between 'image' and 'image_ref'
            ob_val = objective(image, image_ref)
            # ob_val.register_hook(lambda grad: print(grad))
            # image.register_hook(lambda grad: print(grad))
            # Back-propagate errors to input parameters

            ob_val.backward()

            # Optimizer: take a gradient step
            opt.step()
            loss.append(ob_val.item())
            # Compare iterate against ground-truth value
            total_err = 0
            print('Iteration %03i:' % it, end ="")
            for i, (k, v) in enumerate(params_torch.items()):
                err_ref = objective(v, param_ref[i]).item()
                print(' |%s error=%g|' % (k, err_ref), end ="")
                total_err+=err_ref
            print(' |total error=%g| loss=%g' % (total_err, ob_val.item()))
            err.append(total_err)

    time_b = time.time()
    plot_per_epoch('/home/roironen/mitsuba2/projects/dev/grad2grad/test_output/', 'Errors', [loss[:iterations], err[:iterations], loss[iterations:], err[iterations:]], ['L2 loss - measurements', 'L2 loss - parameters','NN L2 loss - measurements', 'NN L2 loss - parameters'], 'log')
    print()
    print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))
