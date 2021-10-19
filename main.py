import numpy as np
import matplotlib.pyplot as plt
import math

# Taylor expansions to approximate derivatives
# Start with ODEs for simplicity

# Forward finite difference for radioactive decay

N0 = 1000                                                                      # Value at t = 0; initial condition
t = np.linspace(0, 10, 100)                                          # Array of t values
dt = t[1] - t[0]
tau = 1.5 


def an_sol(t):                                                                 # Analytical solution for comparison
    n = N0 * math.exp(-t / tau)
    return n

N_approx = np.zeros(len(t))                                           # List of approx. values of N
N_approx[0] = N0                                                               # Initial condition set
N_exact = np.zeros(len(t))                                            # List of exact values of N
N_exact[0] = N0

N_prime = [-N0/tau]                                                            # List of derivatives at each point, N'(t=0) already set

for i in range(1, len(t)):                                                     # Skips first point as initial condition already filled in
    N = N_prime[i-1]*dt + N_approx[i-1]
    N_approx[i] = N                                                            # Calculating and storing next N value
    
    new_N_prime = -N/tau
    N_prime.append(new_N_prime)                                                # Calculating and storing next N' value
    
    N_exact[i] = an_sol(t[i])                                                  # Calculating and storing exact value
    
# Calculating errors
errors = []

for i in range(0, len(t)):
    error = (N_approx[i] - N_exact[i]) / N_exact[i]
    errors.append(error)