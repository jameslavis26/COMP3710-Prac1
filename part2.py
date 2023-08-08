'''
Part 2 of the COMP3710 Demo lab 1. 
Author: James Lavis s4501559
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resolution = 0.00025

Y, X = np.mgrid[-1.3:1.3:resolution, -2:1:resolution]
# Y, X = np.mgrid[-0.5:0.5:resolution, -1.5:-0.5:resolution]

x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)

zs = z.clone()
ns = torch.zeros_like(z)

z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

# Mandlebrot_set
for i in range(200):
    t1 = time.time()
    # Compute z_{i+1} = z_i^2 + c
    zs_ = zs*zs + z
    # Have we diverged?
    not_diverged = torch.abs(zs_) < 4.0 # Any greater and it has diverged
    # Update variables
    ns += not_diverged
    zs = zs_
    t2 = time.time()
    dt = t2 - t1
    print(f"Expected time left {dt*(200 - i):.3f}", end = '\r')


fig = plt.figure(figsize=(16, 10))

def processFractal(a):
    """Display an array of iteration counts as a colorful picture of a fractal. 
       Function taken directly from lab manual. Dont quite understand how it works.
    """
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
# plt.savefig("figures/part2_mandlebrot_hd.png")
plt.show()