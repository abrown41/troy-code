import numpy as np
import matplotlib . pyplot as plt
##input variables##
D = 0.5 # diffusion constant
delta_x = 0.01
delta_t = (delta_x*delta_x) / (2*D) # stability condition
L = 5
boundary_left = 0 
boundary_right = 0

x = np.arange(0, L+delta_x, delta_x)


def initial_condition(x):
    return (np.exp(-(x-2.5)**2))


def nsd(u, delta_x):
    """
    compute the numerical second derivative of u w.r.t x with the three point
    central difference rule.
    """
    d2f = np.zeros(len(x))
    for i in range(1, len(x)-1):
# finite difference rule at each point
        d2f[i] = (u[i+1] + u[i-1] -2*u[i]) 
    d2f = d2f / (delta_x * delta_x)
    return d2f


def update(u, delta_x, delta_t, D):
    """ Compute the change in u over time step delta_t """
    return (D * nsd(u, delta_x) * delta_t)


def time_propagate(T, delta_x, delta_t, D, u):
    """ use FTCS to compute u(x,T)

    Parameters
    ----------
    T : float
        final time for time propagation
    delta_x : float
        spatial grid spacing
    delta_t : float
        time step
    D : float
        diffusion constant
    u : array-like
        initial condition for the diffusion constant

    Returns
    -------
    u : array-like
        solution of diffusion equation u(x,t=T)
    """

    t = 0
    while t < T:
        delta_u = update(u, delta_x, delta_t, D)
        u += delta_u
        t += delta_t
        u[0] = boundary_left
        u[-1] = boundary_right
    return u

u = initial_condition(x)
plt.plot(x,initial_condition(x), label = "initial")
plt.plot(x,time_propagate(1.0, delta_x, delta_t, D, u), label = "T=1.0")
plt.legend()
plt.show()
