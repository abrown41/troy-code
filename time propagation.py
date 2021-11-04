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
dt = 0.01*(dx**2) / abs(D)
psi_0 = 0 + 0j
psi_L = 0 + 0j                # Boundary conditions

x = np.arange(0, L+dx, dx) # Discretising domain

def initial_condition(x):                       
    return (np.sqrt(2/L)*np.sin(np.pi * x))

def nsd(psi, dx):
    d2f = np.zeros(len(x))
    d2f = d2f.astype('complex64')
    for i in range(1, len(x)-1):
        d2f[i] = (psi[i+1] + psi[i-1] -2*psi[i])
    d2f = d2f/(dx**2)
    return d2f

def update(psi, dx, dt, D):
    return (D * nsd(psi, dx) * dt)

def time_propagate(T, dx, dt, D, psi):
    t = 0
    while t<T:
        d_psi = update(psi*np.exp((-1j * np.pi**2 * t)/hbar), dx, dt, D)
        psi += d_psi
        t += dt
        psi[0] = psi_0
        psi[-1] = psi_L
    return psi

psi = initial_condition(x) + 0j*initial_condition(x)

for ii in range(0,11):
    psi = time_propagate(ii*100*dt, dx, dt, D, psi) # propagate for 100 time steps, then stop and plot
    plt.plot(x,np.abs(psi)**2, label ="T="+str(ii*100)+"dt") # plot probability
plt.legend()