'''
Part 3 of the COMP3710 Demo lab 1. 
Author: James Lavis s4501559
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
from time import time

print("Start")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resolution = 100000

l1 = torch.linspace(0, 3.4, 10000)
l2 = torch.linspace(3.4, 3.5, 10000)
l3 = torch.linspace(3.5, 3.7, 50000)
l4 = torch.linspace(3.7, 3.8, 10000)
# l5 = torch.linspace(3.8, 4.2, 10000)
l = torch.concatenate([l1, l2, l3, l4, l5])
X = torch.rand(size=l.size())

X.to(device)

print("Data Prepared")

def compute_fractal_portion(X, l):
    for t in range(100):
        X = l*X*(1-X)
    return X

def compute_fractal(X, l):
    cores = 8
    lz = X.size()[0]
    dx = int(lz/cores)

    pool = Pool(processes=cores)
    # X = pool.starmap(
    #     compute_fractal_portion, # function to call
    #     [[X[i*dx:(i+1)*dx], l[i*dx:(i+1)*dx]] for i in range(cores+1)] # iterable of arguments
    # )

    X = pool.starmap(
        compute_fractal_portion, # function to call
        [[X[i], l[i]] for i in range(lz)] # iterable of arguments
    )

    X = torch.Tensor(X)
    return X

X = compute_fractal(X, l).cpu().numpy()
l = l.cpu().numpy()

fig = plt.figure(figsize=(16, 10))

plt.plot(l, X, ',k')

plt.savefig("figures/part3_logistic_hd.png")
plt.show()