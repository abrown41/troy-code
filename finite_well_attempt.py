# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 20:24:20 2021

@author: Troy
"""

import numpy as np
import matplotlib.pyplot as plt

# Particle in a box
# Atomic units so numbers stay in sensible range

hbar = 1
m = 1
L = 1                    # Atomic units for each
D = (1j * hbar)/(2 * m)  # Coefficient on second spatial derivative, named to mirror diffusion code
V0 = float(input('Potential?'))
dx = 0.01
dt = 0.01*(dx**2) / abs(D)
x = np.linspace(-L, L, 101)                                                    # Discretising domain
dx = x[1] - x[0]                                                               # Defining grid spacing
psi_left = 0 + 0j                                                              # Chosen to approximate psi asymptote
psi_right = 0 + 0j

def initial_condition(x):                       # Simple Gaussian function, corresponding to particle's position being roughly known
    return (np.exp((-(x)**2)/0.001))

def nsd(psi, dx):
    d2f = np.zeros(len(x))
    d2f = d2f.astype('complex64')
    for i in range(1, len(x)-1):
        d2f[i] = (psi[i+1] + psi[i-1] -2*psi[i])
    d2f = d2f/(dx**2)
    return d2f

def update(psi, dx, dt, D):
    d_psi = []
    for i in range(0, len(x)):
        if -L/2 < x[i] < L/2:
            d_psi_i = D * nsd(psi, dx)[i] * dt
            d_psi.append(d_psi_i)
        else:
            d_psi_i = ((D * nsd(psi, dx)[i]) - ((1j/2*m) * V0 * psi[i]))*dt
            d_psi.append(d_psi_i)
    return d_psi

def time_propagate(T, dx, dt, D, psi):
    t = 0
    while t<T:
        d_psi = update(psi, dx, dt, D)
        psi += d_psi
        t += dt
        psi[0] = psi_left
        psi[-1] = psi_right
    return psi

psi = initial_condition(x) + 0j*initial_condition(x)

for ii in range(0,11):
    psi = time_propagate(ii*100*dt, dx, dt, D, psi) # propagate for 100 time steps, then stop and plot
    plt.plot(x,np.abs(psi)**2, label ="T="+str(ii*100)+"dt") # plot probability
plt.legend()