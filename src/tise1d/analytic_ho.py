import torch
from scipy.special import factorial, hermite
import numpy as np
from numpy.typing import NDArray

hbar = 1
m = 1
omega = 1

def psi_analytic(x: NDArray[np.floating], n: int):
    alpha = m * omega / hbar
    return (1/(np.sqrt((2**n)*factorial(n))*hbar*(alpha*np.pi)**0.25))*np.exp(-alpha*x**2/2)*hermite(n)(np.sqrt(alpha)*x)

def energy_analytic(n: int):
    return hbar*omega*(n + 0.5)