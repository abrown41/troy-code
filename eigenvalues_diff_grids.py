# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 21:10:39 2021

@author: Troy
"""

import numpy as np
import matplotlib.pyplot as plt

# Particle in a box
# Atomic units so numbers stay in sensible range

L = 1                                                                          # Width of box in Bohr radii

for m in [1, 2, 5, 10]:
    x = np.linspace(0, L, 100*m + 1)                                           # Discretising domain
    dx = x[1] - x[0]                                                           # Defining grid spacing
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
         del energies_list[0]                                                  # Nonsense energies both turned out to be negative while all others were positive
    
    eigenstates = np.arange(1, 100)
        
    plt.plot(eigenstates, energies_list[:99], label=f'{100*m} grid points')

an_nrg = eigenstates**2 * np.pi**2 / 2                                         # Analytical energy values

plt.plot(eigenstates, an_nrg, label='Analytical')    
plt.legend()
plt.title('Approximate and analytical energy values as a function of n')
plt.xlabel('n')
plt.ylabel('Energy (Ha)')