# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 14:34:09 2022

@author: Troy
"""

import numpy as np
import matplotlib.pyplot as plt

dx = 0.1
dt = 1e-6
L = 20
x = np.arange(dx, L+dx, dx)                                                    # Discretising domain, excluding origin to avoid division by 0
V = -1/ (x)
hbar = 1
T = 0.1

def ground(x):
    return 1/np.sqrt(np.pi) * np.exp(-x)

# Set up matrix problem
A = np.zeros((len(x), len(x)))
A[0, 0] = (1/dx**2) - 1/(x[1])
A[0, 1] = -1/(2*(dx**2))
A[-1, -1] = (1/dx**2) - 1/(x[-2])
A[-1, -2] = -1/(2*(dx**2))

for i in range(1, len(x)-1):
    A[i, i-1] = -1/(2*(dx**2))
    A[i, i] = (1/dx**2) - 1/(x[i])
    A[i, i+1] = -1/(2*(dx**2))

eig_val, eig_vec = np.linalg.eig(A)

sorted_eig_val = np.sort(eig_val)
sorted_eig_vec = eig_vec[:, eig_val.argsort()]

# for i in range(0, 5):                                                          # Plots first few eigenfunctions
#     eigen_energy = sorted_eig_val[i]
#     col_loc = eig_val.tolist().index(eigen_energy)                             # Locates the column in the array of eigenvectors corresponding to each eigenstate
#     eigenfunction = eig_vec[:, col_loc]
#     if eigenfunction[0] > eigenfunction[1]:                                    # Some eigenvectors have physically unimportant sign change; this flips such cases to compare to analytical solution 
#         eigenfunction *= -1  
#     plt.plot(x, eigenfunction, label = f'n = {i-1}')
#     plot_title = 'First five eigenstates'
#     plt.title(plot_title)
#     plt.xlabel('x')
#     plt.ylabel(r'$\Psi$')

## Initialise system in ground state ##

psi = sorted_eig_vec[:, 0] + 0j*sorted_eig_vec[:, 0]

# initial_coefficients = []
# for j in range(0, len(eig_val)):
#     initial_coefficients.append(np.trapz((eig_vec[:, column_index] - 0j*eig_vec[:, column_index])*eig_vec[:, j] + 0j*eig_vec[:, j]))

# plt.plot(np.arange(0, len(eig_val)), initial_coefficients)

E0 = 0.5
omega = 2*np.pi/(T)

def nsd(psi, dx):
    d2f = np.zeros(len(x))
    d2f = d2f.astype('complex64')
    for i in range(1, len(x)-1):
        d2f[i] = (psi[i+1] + psi[i-1] -2*psi[i])
    d2f = d2f/(dx**2)
    return d2f

def external_potential(t):
    return E0*np.sin(omega*t)*x

def update(psi, dx, dt, t):
    d_psi = (-0.5*nsd(psi, dx) + (V+external_potential(t))*psi)* -1j * dt
    return d_psi

def time_propagate(T, dx, dt, psi):
    t = 0
    while t<T:
        d_psi = update(psi, dx, dt, t)
        psi += d_psi
        t += dt
        # psi[0] = psi_left
        # psi[-1] = psi_right
    return psi



for ii in range(0, 11):
    psi = time_propagate(ii*(T/10), dx, dt, psi) # propagate for 100 time steps, then stop and plot
    plt.figure(0)
    plt.plot(x,np.abs(psi)**2, label = f't={ii*T/10:.4f}') # plot probability
    plt.figure(2)
    plt.plot(x, V+external_potential(ii*(T/10)), label = f't={ii*T/10:.4f}')
plt.figure(0)
plt.legend()
plt.title('Ground state propagated in time')

plt.figure(2)
plt.title('Total field')
plt.legend()
plt.show()