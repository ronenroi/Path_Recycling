# Simple inverse rendering example: render a cornell box reference image,
# then replace one of the scene parameters and try to recover it using
# differentiable rendering and gradient-based optimization. (PyTorch)

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, Vector3f
from mitsuba.core.xml import load_file
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import  write_bitmap
from projects.dev.grad2grad.my_render import render_torch


import torch
import time

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])

Thread.thread().file_resolver().append('cbox')
scene = load_file('/home/roironen/mitsuba2/resources/data/docs/scenes/variants_cbox_rgb.xml')

# Find differentiable scene parameters
params = traverse(scene)

# Discard all parameters except for one we want to differentiate
params.keep(['red.reflectance.value','green.reflectance.value'])
#params.keep(['red.reflectance.value'])


# Print the current value and keep a backup copy
param_ref = []
for k,_ in params.properties.items():
    param_ref.append(params[k].torch())
print(param_ref)

# Render a reference image (no derivatives used yet)
image_ref = render_torch(scene, spp=80)
crop_size = scene.sensors()[0].film().crop_size()
write_bitmap('/home/roironen/mitsuba2/projects/dev/grad2grad/output/out_ref.png', image_ref, crop_size)

# Change the red and green walls into a bright white surface
for k,_ in params.properties.items():
    params[k] = [.9, .9, .9]
params.update()

# Which parameters should be exposed to the PyTorch optimizer?
params_torch = params.torch()

# Construct a PyTorch Adam optimizer that will adjust 'params_torch'
opt = torch.optim.Adam(params_torch.values(), lr=.2)
objective = torch.nn.MSELoss()

time_a = time.time()
spp = 10
iterations = 100

def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, x, grad_outputs = grad_outputs, retain_graph=True)[0]
    return grad

def jacobian(y, x):
    """Compute dy/dx = dy/dx @ grad_outputs;
    for grad_outputs in [1, 0, ..., 0], [0, 1, 0, ..., 0], ...., [0, ..., 0, 1]"""
    jac = torch.zeros(y.shape[0], x.shape[1])
    for i in range(y.shape[0]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[i] = 1
        jac[i] = gradient(y, x, grad_outputs = grad_outputs)
    return jac


for it in range(iterations):
    # Zero out gradients before each iteration
    opt.zero_grad()

    # Perform a differentiable rendering of the scene
    image = render_torch(scene, params=params, unbiased=True,
                         spp=10, **params_torch)
    image.retain_grad()
    write_bitmap('/home/roironen/mitsuba2/projects/dev/grad2grad/output/out_%03i.png' % it, image, crop_size)

    # Objective: MSE between 'image' and 'image_ref'
    ob_val = objective(image, image_ref)
    #ob_val.register_hook(lambda grad: print(grad))
    #image.register_hook(lambda grad: print(grad))
    # Back-propagate errors to input parameters

    ob_val.backward(retain_graph=True, create_graph=True)
    jac = jacobian(image,params_torch['red.reflectance.value'])

    grad = image.grad
    getBack(ob_val.grad_fn)

    # Optimizer: take a gradient step
    opt.step()

    # Compare iterate against ground-truth value
    #err_ref = objective(params_torch['red.reflectance.value'], param_ref)
    #print('Iteration %03i: error=%g' % (it, err_ref), end='\r')

time_b = time.time()

print()
print('%f ms per iteration' % (((time_b - time_a) * 1000) / iterations))
