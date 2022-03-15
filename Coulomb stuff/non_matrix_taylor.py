# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 13:40:43 2022

@author: Troy
"""

import numpy as np
from scipy.linalg.blas import zgemv
import matplotlib.pyplot as plt
import sys

dt = 0.001
dx = 0.1
order = 1
t_initial = 0
T = 6
N = int(T/dt) # number of time steps
L = 40 # extent of domain ±L
x = np.arange(0, L+dx, dx)
# x = np.arange(dx, L+dx, dx)
# Softened core potential: mimics coulomb but removes singularity at x=0
q = 4
V = - 1/(x**q+0.1**q)**(1/q) 
# V = -1/np.abs(x)

A = np.zeros((len(x), len(x)))

for i in range(1, len(x)-1):
    A[i, i-1] = -1/(2*(dx**2))
    A[i, i] = (1/dx**2) + V[i]
    A[i, i+1] = -1/(2*(dx**2))

eig_val, eig_vec = np.linalg.eig(A)

sorted_eig_val = np.sort(eig_val)
sorted_eig_vec = eig_vec[:, eig_val.argsort()]



# Set up matrix problem
def init(sorted_eig_vec):
    psi = sorted_eig_vec[:, 0] + 0j*sorted_eig_vec[:, 0]
    return(psi)

E0 = 0.5
omega = 1  # 2*np.pi/(T)

def external_potential(t):
    return E0*np.sin(omega*t)*np.sin(np.pi*t/T)**8 * x

def nsd(psi, dx):
    return np.concatenate(([0], (psi[2:] + psi[:-2] - 2*psi[1:-1])/(dx*dx), [0]))

def update(psi, dx, dt, t):
    d_psi = (-0.5*nsd(psi, dx) + (V+external_potential(t))*psi)* -1j * dt
    return d_psi

def taylor(psi, dt, t, order):
    current_term = psi
    for ii in range(order):
        alpha=(1.0/(ii+1))*-1j*dt
        next_term = (-0.5*nsd(current_term, dx) + (V+external_potential(t))*current_term)*alpha 
        psi += next_term
        current_term = next_term
    return(psi)

def FTCS_propagate(T, dx, dt, psi):
    t = 0
    while t<T:
        d_psi = update(psi, dx, dt, t)
        psi += d_psi
        norm = np.trapz(np.abs(psi)**2)
        if norm > 2:
            print(f"ERROR FTCS: norm = {norm}")
            break
        t += dt
    return psi

def taylor_propagate(T, dt, psi, order=1):
    t = 0
    while t<T:
        psi = taylor(psi, dt, t, order)
        t += dt
        norm = np.trapz(np.abs(psi)**2)
        if norm > 2:
            print(f"ERROR Taylor: norm = {norm}")
            break
    return psi

ground_norm = np.trapz(np.abs(sorted_eig_vec[:, 0] + 0j*sorted_eig_vec[:, 0])**2)
norms = [ground_norm]

psi = init(sorted_eig_vec)
psi = taylor_propagate(T, dt, psi, order=4)
print(psi[1])
plt.plot(np.real(psi), 'r', label="taylor -real")
plt.plot(np.imag(psi), 'r--', label="taylor -imag")
psi= init(sorted_eig_vec)
ftcsmeth = FTCS_propagate(T, dx, dt, psi)
print(psi[1])
plt.plot(np.real(ftcsmeth), 'b--', label="FTCS -real")
plt.plot(np.imag(ftcsmeth), 'b', label="FTCS -imag")

plt.legend()
#plt.plot(range(len(norms)), norms)
plt.show()
