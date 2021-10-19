import numpy as np
import matplotlib.pyplot as plt
import math

# Particle in a box
# Atomic units so numbers stay in sensible range

L = 1                                                                          # Width of box in Bohr radii
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
    
eigenstates = np.arange(1, len(x)-1)
an_nrg = eigenstates**2 * np.pi**2 / 2                                         # Analytical energy values

for i in range(2, 7):                                                          # Plots first few eigenfunctions, disregarding two trivial solutions
    eigen_energy = sorted_eig_val[i]
    col_loc = eig_val.tolist().index(eigen_energy)                             # Locates the column in the array of eigenvectors corresponding to each eigenstate
    eigenfunction = eig_vec[:, col_loc]
    if eigenfunction[0, :] * eigenfunction[1, :] < 0:                          # Some eigenvectors have physically unimportant sign change; this flips such cases to compare to analytical solution 
        eigenfunction *= -1                                                    
    plt.figure(i)
    plt.plot(x, eigenfunction)
    plot_title = 'n = ' + str(i-1)
    plt.title(plot_title)


plt.plot(eigenstates, energies_list, label='Approximate eigen-energies')
plt.plot(eigenstates, an_nrg, label='Analytical eigen-energies')
plt.title('Energies as a function of n')
plt.legend()
plt.xlabel('n')
plt.ylabel('Energy (Ha)')

plt.figure(100)
plt.plot(x, exact)
plt.title('Analytical Solution')