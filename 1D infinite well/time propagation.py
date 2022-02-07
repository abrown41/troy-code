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
dx = float(input('Spatial grid spacing: '))
dt = float(input('Temporal grid spacing: '))
psi_0 = 0 + 0j
psi_L = 0 + 0j                # Boundary conditions
T = 0.002
scheme = input('Finite difference scheme: ')

x = np.arange(0, L+dx, dx) # Discretising domain

def initial_condition(x):                       
    return (np.exp((-(x-0.5)**2)/0.001))


if scheme == '3-point':
    def nsd(psi, dx):
        d2f = np.zeros(len(x))
        d2f = d2f.astype('complex64')
        for i in range(1, len(x)-1):
            d2f[i] = (psi[i+1] + psi[i-1] -2*psi[i])
        d2f = d2f/(dx**2)
        return d2f
elif scheme == '5-point':
    def nsd(psi, dx):
        d2f = np.zeros(len(x))
        d2f = d2f.astype('complex64')
        for i in range(2, len(x)-2):
            d2f[i] = (-1/12 * psi[i-2]) + (4/3 * psi[i-1]) + (-2.5 * psi[i]) + (4/3 * psi[i+1]) + (-1/12 * psi[i+2])
        d2f = d2f/(dx**2)
        return d2f
else:
    print('Please enter 3-point or 5-point exactly as written.')

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

initial_norm = np.linalg.norm(psi)
psi /= initial_norm
# squared_norm = np.trapz(np.abs(psi)**2, x)
# psi /= squared_norm

norms = []
times=[]

for ii in range(0, 11):
    squared_norm = np.trapz(np.abs(psi)**2, x)
    print(squared_norm)
    norms.append(squared_norm)
    psi = time_propagate(ii*(T/10), dx, dt, D, psi) # propagate for 100 time steps, then stop and plot
    # plt.plot(x,np.abs(psi)**2, label ="T="+str(ii*100)+"dt") # plot probability
    # plt.figure(ii)
    # plt.plot(x, np.real(psi), label = r'Re($\Psi$)')
    # plt.plot(x, np.imag(psi), label = r'Im($\Psi$)')
    # plt.title(fr'Components of $\Psi$ at t={ii*100*dt:.4f}')
    # plt.legend()
    times.append(ii*(T/10))

print(f'Average norm across all time steps: {np.mean(norms)}')
# plt.legend()
# plt.title(r'$|\Psi|^2$ at various points in time')

# plt.figure(30)
# plt.plot(range(0,11), norms)
# plt.title(r'Relative squared norm of $\Psi$ at each time')
# plt.ylim([0.8, 1.2])
# plt.xlabel(f'Time (multiples of {T/10})')
# plt.ylabel(r'Relative squared norm of $\Psi$')