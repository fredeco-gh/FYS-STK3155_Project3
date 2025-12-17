import numpy as np
from scipy.special import hermite,factorial

def potential(x): 
    return 0.5*x**2

def solve_schrodinger(x,E,L): 
    A = np.zeros((len(x) - 2,len(x) - 2))
    n = A.shape[0]
    delta_x = x[1] - x[0]
    A[0,0] = 1/delta_x**2 + potential(x[0] + delta_x)
    A[0,1] = -0.5/delta_x**2
    for i in range(1,n - 1): 
        A[i,i-1] = -0.5/delta_x**2
        A[i,i] = 1/delta_x**2 + potential(x[0] + delta_x*(i+1))
        A[i,i+1] = -0.5/delta_x**2
    A[n-1,n-2] = -0.5/delta_x**2
    A[n-1,n-1] = 1/delta_x**2 + potential(x[-1] - delta_x)

    eigval, psi = np.linalg.eigh(A)
    i = np.argmin(np.abs(eigval - E))
    psi_num = np.zeros_like(x)
    psi_num[1:n+1] = psi[:,i]
    return psi_num/np.sqrt(2*L*np.mean(psi_num**2))

