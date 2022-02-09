# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 14:34:09 2022

@author: Troy
"""

import numpy as np
from scipy.linalg.blas import zgemv
import matplotlib.pyplot as plt
import sys

dt = 0.0001
dx = 0.1
order = 6
T = 1
N = int(T/dt) # number of time steps
L = 20 # extent of domain ±L
x = np.arange(-L, L+dx, dx)
# Softened core potential: mimics coulomb but removes singularity at x=0
q = 2
V = - 1/(x**q+0.1**q)**(1/q)


def ground(x):
    return 1/np.sqrt(np.pi) * np.exp(-x)


# Set up matrix problem
A = np.zeros((len(x), len(x)))
A[0, 0] = (1/dx**2) + V[0]
A[0, 1] = -1/(2*(dx**2))
A[-1, -1] = (1/dx**2) + V[-1]
A[-1, -2] = -1/(2*(dx**2))

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


def Taylor(H, psi, dt, order=8):
    """ Compute the application of the Hamiltonian, H, to the wavefunction, psi,
    approximated using a Taylor expansion of the exponential operator."""
    Hpsi = 1.0*psi
    for ii in range(order):
        Hpsi += (-1j*dt/(ii+1))*zgemv(1.0, H, Hpsi)
    return Hpsi


def nsd(psi, dx):
    d2f = np.zeros(len(x))
    d2f = d2f.astype('complex64')
    for i in range(1, len(x)-1):
        d2f[i] = (psi[i+1] + psi[i-1] - 2*psi[i])
    d2f = d2f/(dx**2)
    return d2f


def external_potential(t):
    return E0*np.sin(omega*t)*x


def update(psi, dx, dt, t):
    d_psi = (-0.5*nsd(psi, dx) + (V+external_potential(t))*psi) * -1j * dt
    return d_psi


def time_propagate(T, dx, dt, psi):
    t = 0
    while t < T:
        psi = Taylor(A, psi, dt, order=order)
        t += dt
        psi[0] = 0
        psi[-1] = 0
    return psi


plt.figure(0)
#plt.plot(x, np.abs(psi)**2,  label='t=0')  # plot probability
plt.plot(x, np.real(psi),  label='t=0')  # plot probability
plt.plot(x, np.imag(psi),  label='t=0')  # plot probability

psi = time_propagate(N*dt, dx, dt, psi)

#plt.plot(x, np.abs(psi)**2,  label=f't={N*dt}')  # plot probability
plt.plot(x, np.real(psi), label=f't={N*dt}')  # plot probability
plt.plot(x, np.imag(psi), label=f't={N*dt}')  # plot probability
plt.figure(0)
plt.legend()
plt.title('Ground state propagated in time')
plt.show()
# plt.savefig(f"dt_{dt}_dx_{dx}_order_{order}.png")

# plt.figure(2)
#plt.title('Total field')
# plt.legend()
