# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:23:35 2021

@author: Troy
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# Particle in a box
# Atomic units so numbers stay in sensible range

L = 1                                                                          # Width of box
n = int(input('Eigenstate? '))                                                 # Lowest eigenstate
k = (n * math.pi / L)
norm = math.sqrt((2/L))                                                        # Normalisation coefficient
hbar = 1                                                                       # Reduced Planck constant
m = 1                                                                          # Electron mass
E = (n**2 * math.pi**2 * hbar**2) / (2 * m * L**2)                             # Energy eigenvalue

x = np.linspace(0, L, 101)                                                     # Discretising domain
dx = x[1] - x[0]                                                               # Defining grid spacing

def wavefunction(x):                                                           # Analytical solution for comparison
    psi = norm*np.sin(k*x)
    return psi

exact = wavefunction(x)

# Construct matrix equation Ab_n = E_n * b_n, where b is vector containing unknowns and E are eigenvalues
# Use np.linalg.eig to find b_n and E_n

A = np.zeros((len(x), len(x)))
A[0, 0] = 1
A[-1, -1] = 1

for i in range(1, len(x)-1):
    A[i, i-1] = 1
    A[i, i] = -2
    A[i, i+1] = 1
A = (1/dx**2) * A

eig_val, eig_vec = np.linalg.eig(A)

# Sort eigenvalues in ascending order and extract energy values; Matrix A satisfies psi'' = -2 * E * psi. Actual energies are thus eigenvalues/-2
sorted_eig_val = np.sort(eig_val)[::-1]
energies = sorted_eig_val / -2

# Two of the calculated energies come from the trivial solution psi(0)=psi(L)=0 and yield nonsense plots; remove them
energies_list = energies.tolist()
while energies_list[0] < 0:
    del energies_list[0]                                                       # Nonsense energies both turned out to be negative while all others were positive
    
eigenstates = np.arange(1, 100)
an_nrg = eigenstates**2 * np.pi**2 / 2                                         # Analytical energy values

# Eigenvectors founds by linalg.eig have physically unimportant sign change due to construction of the problem; I flip signs to compare to analytical solution
eig_vec *= -1                                                                  


for i in range(2, 7):                                                          # Plots first few eigenfunctions, disregarding two trivial solutions
    plt.figure(i)
    plt.plot(x, eig_vec[:, eig_val.tolist().index(sorted_eig_val[i])])
    plot_title = 'n = ' + str(i-1)
    plt.title(plot_title)


plt.plot(eigenstates, energies_list, label='Approximate eigen-energies')
plt.plot(eigenstates, an_nrg, label='Analytical eigen-energies')
plt.title('Energies as a function of n')
plt.legend()