# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:50:21 2021

@author: Troy
"""

import numpy as np
import matplotlib.pyplot as plt

# Define variables

hbar = 1
m = 1
L = 1                    # Atomic units for each
D = (1j * hbar)/(2 * m)  # Coefficient on second spatial derivative, named to mirror diffusion code
dx = 0.01
dt = (dx**2) / abs(D)
psi_0 = 0
psi_L = 0                # Boundary conditions

x = np.arange(0, L+dx, dx) # Discretising domain

def initial_condition(x):                       # Simple Gaussian function, corresponding to particle's position being roughly known
    return (np.exp((-(x-0.5)**2)/0.2))

def nsd(psi, dx):
    d2f = np.zeros(len(x))
    for i in range(1, len(x)-1):
        d2f[i] = (psi[i+1] + psi[i-1] -2*psi[i])
    d2f = d2f/(dx**2)
    return d2f

def update(psi, dx, dt, D):
    return (D * nsd(psi, dx) * dt)

def time_propagate(T, dx, dt, D, psi):
    psi = psi.astype('complex64')
    t = 0
    while t<T:
        d_psi = update(psi, dx, dt, D)
        psi += d_psi
        t += dt
        psi[0] = psi_0
        psi[-1] = psi_L
    return psi

psi = initial_condition(x)
psi_final = time_propagate(10, dx, dt, D, psi)

plt.plot(x, psi, label = 'Initial')
plt.ylim([0, np.amax(psi)*1.1])
plt.plot(x, psi_final, label = 'T = 1.0')
plt.legend()