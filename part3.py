'''
Part 3 of the COMP3710 Demo lab 1. 
Author: James Lavis s4501559
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.multiprocessing import Pool
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

w = 2/3
N_max = 60

x = torch.linspace(0, 1, 100, dtype=torch.float64)
y = torch.zeros_like(x)

def triangle(x):
    k = torch.round(x)
    return torch.abs(x - k)

for n in range(N_max):
    y = (w**n)*triangle((2**n)*x)

# y = triangle(x)

plt.plot(x, y)
plt.show()
