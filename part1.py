'''
Part 1 of the COMP3710 Demo lab 1. 
Author: James Lavis s4501559
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

x = torch.Tensor(X).to(device)
y = torch.Tensor(Y).to(device)

z_exp = torch.exp(-(x**2 + y**2)/2)
z_cos = torch.cos(x + y)
z = z_exp*z_cos

plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()