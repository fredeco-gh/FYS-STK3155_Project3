import torch
from scipy.special import factorial, hermite
import numpy as np
from numpy.typing import NDArray
from tise1d import tise1d
import matplotlib.pyplot as plt

hbar = 1
m = 1
omega = 1

# Analytical solution to n-th quantum harmonic oscillator (QHO) wavefunction
def psi_analytic(x: NDArray[np.floating], n: int):
    alpha = m * omega / hbar
    return (1/(np.sqrt((2**n)*factorial(n))*hbar*(alpha*np.pi)**0.25))*np.exp(-alpha*x**2/2)*hermite(n)(np.sqrt(alpha)*x)

# Analytical formula for the n-th energy eigenvalue
def energy_analytic(n: int):
    return hbar*omega*(n + 0.5)

# Plot and compare predicted (by the neural network) and exact QHO wavefunction 
def compare_analytic(x: torch.Tensor, x_lim: tuple[float, float], pinn: tise1d.PINN, n: int, show=True):
    # Sort for plotting
    order = x.argsort(dim=0).squeeze()
    x = x[order]
    
    with torch.no_grad():
        psi_pred = pinn(x).detach().cpu().numpy()

    x_np = x.detach().cpu().numpy() # Convert to numpy array for analytic function
    psi_exact = psi_analytic(x_np, n=n)

    # Flip sign if negative overlap
    overlap = np.sum(psi_pred*psi_exact)
    if overlap < 0: 
        psi_pred = -psi_pred
 
    energy_exact = energy_analytic(n)
    energy_pred = pinn.E.detach().cpu().item()
    energy_rel_error = abs(energy_pred-energy_exact)/energy_exact   # relative error of predicted energy relative to the analytical one

    print(f"Predicted Energy: {energy_pred:.6f}, Exact Energy: {energy_exact:.6f}, Relative Error: {energy_rel_error:.6%}")
    
    L = x_lim[1] - x_lim[0]
    psi_pred_normalized = psi_pred/np.sqrt(L*np.mean(psi_pred**2)) # Normalize effectively using monte carlo integration
    
    plt.plot(x_np, psi_pred_normalized, label='PINN Prediction')
    plt.plot(x_np, psi_exact, label='Analytic Solution', linestyle='dashed')
    plt.xlabel('x')
    plt.ylabel('ψ(x)')
    plt.title('Comparison of PINN and Analytic Solution')
    plt.legend()
    if show:
        plt.show()