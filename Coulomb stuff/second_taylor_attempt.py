# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:40:43 2022

@author: Troy
"""

import numpy as np
from scipy.linalg.blas import zgemv
import matplotlib.pyplot as plt
import sys

dt = 0.0001
dx = 0.1
order = 1
t_initial = 0
T = 1
N = int(T/dt) # number of time steps
L = 40 # extent of domain ±L
x = np.arange(0, L+dx, dx)
# x = np.arange(dx, L+dx, dx)
# Softened core potential: mimics coulomb but removes singularity at x=0
q = 4
V = - 1/(x**q+0.1**q)**(1/q)
# V = -1/np.abs(x)




# Set up matrix problem
A = np.zeros((len(x), len(x)))
A[0, 0] = (1/dx**2) + V[0]
# A[0, 1] = -1/(2*(dx**2))
A[-1, -1] = (1/dx**2) + V[-1]
# A[-1, -2] = -1/(2*(dx**2))

for i in range(1, len(x)-1):
    A[i, i-1] = -1/(2*(dx**2))
    A[i, i] = (1/dx**2) + V[i]
    A[i, i+1] = -1/(2*(dx**2))

eig_val, eig_vec = np.linalg.eig(A)

sorted_eig_val = np.sort(eig_val)
sorted_eig_vec = eig_vec[:, eig_val.argsort()]

psi = sorted_eig_vec[:, 0] + 0j*sorted_eig_vec[:, 0]

E0 = 0.5
omega = 1  # 2*np.pi/(T)

def external_potential(t):
    return E0*np.sin(omega*t)*x

def test_taylor(H, psi, dt, t, order):
    H_t = H
    laser = external_potential(t)
    for i in range(0, len(x)):
        H_t[i,i] += laser[i]
    current_term = psi
    for ii in range(order):
        next_term = (1/(ii+1))*(-1j*dt)*zgemv(1, H_t, current_term)
        psi += next_term
        current_term = next_term
    return psi

t = 0
ground_norm = np.trapz(np.abs(sorted_eig_vec[:, 0] + 0j*sorted_eig_vec[:, 0])**2)
norms = [ground_norm]

while t < T:
    psi = test_taylor(A, psi, dt, t, 8)
    norm = np.trapz(np.abs(psi)**2)
    norms.append(norm)
    # print(norm)
    if norm > 2:
        break
    t += dt
plt.plot(range(len(norms)), norms)