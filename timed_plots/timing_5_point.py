# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 22:44:20 2021

@author: Troy
"""

import timeit

N = int(input('Iterations: '))

setup_code = '''
import numpy as np
import matplotlib.pyplot as plt

iteration = 1
hbar = 1
m = 1
L = 1                    # Atomic units for each
D = (1j * hbar)/(2 * m)  # Coefficient on second spatial derivative, named to mirror diffusion code
V0 = 1000
dx = float(input('Enter spatial grid spacing: '))
dt = float(input('Enter time step: '))
x = np.arange(-L, L+dx, dx)                                                    # Discretising domain
psi_left = 0 + 0j                                                              # Chosen to approximate psi asymptote
psi_right = 0 + 0j
V = np.zeros(len(x))
T = 0.002

for i in range(0, len(x)):
    if x[i] < -L/4 or L/4 < x[i]:
        V[i] = V0

def initial_condition(x):                       # Simple Gaussian function, corresponding to particle's position being roughly known
    return (np.exp((-(x)**2)/0.001))

def nsd(psi, dx):
    d2f = np.zeros(len(x))
    d2f = d2f.astype('complex64')
    for i in range(2, len(x)-2):
        d2f[i] = (-1/12 * psi[i-2]) + (4/3 * psi[i-1]) + (-2.5 * psi[i]) + (4/3 * psi[i+1]) + (-1/12 * psi[i+2])
    d2f = d2f/(dx**2)
    return d2f

def update(psi, dx, dt, D):
    d_psi = (D * nsd(psi, dx) - 1j/hbar * V * psi)* dt
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
'''
program = '''
psi = initial_condition(x) + 0j*initial_condition(x)

norms = []
plt.figure(iteration)
for ii in range(0, 11):
    psi = time_propagate(ii*(T/10), dx, dt, D, psi) # propagate for 100 time steps, then stop and plot
    # norms.append(np.linalg.norm(psi))
    plt.plot(x,np.abs(psi)**2, label ="T="+str(ii*100)+"dt") # plot probability
    # plt.figure(ii)
    # plt.plot(x, np.real(psi), label = r'Re($\Psi$)')
    # plt.plot(x, np.imag(psi), label = r'Im($\Psi$)')
    # plt.legend()
    # plt.title(f'T = {ii*(T/10)}')

plt.legend()
plt.title(f'V = {V0} Ha')

iteration += 1
'''

print(f"Average execution: {timeit.timeit(stmt= program, setup = setup_code, number=N) / N} seconds")