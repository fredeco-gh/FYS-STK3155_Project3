from __future__ import annotations
from typing import Callable
from core.interfaces import PhysicsLoss, PhysicsInformedNN, Potential, AnsatzFactor
import torch
import torch.nn as nn


class Schrodinger1DTimeIndependentPINN(PhysicsInformedNN):
    def __init__(self, model: nn.Module, ansatz_factor: AnsatzFactor,L: float = 1.0, E: float = 0.5):
        super().__init__(model,ansatz_factor)
        self.L = L
        self.E = nn.Parameter(torch.tensor(E,dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."

        raw = self.model(inputs)
        psi = raw*self.ansatz_factor(inputs,self)

             #raw = torch.complex(raw[0], raw[1])

        return psi
    


class LossTISE1D(PhysicsLoss):
    def __init__(self, potential: "Potential" | None) -> None:
        super().__init__()
        self.potential = potential

    """
    Physics loss for the 1D time-independent Schrödinger equation, in units where hbar/2m = 1.
    Potential V(x) is assumed to be 0 everywhere.
    """
    def __call__(self, pinn: "Schrodinger1DTimeIndependentPINN", inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."

        x = inputs.clone().detach().requires_grad_(True)

        psi = pinn(x)

        dpsi_dx = torch.autograd.grad(
            psi,
            x,
            grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0]

        d2psi_dx2 = torch.autograd.grad(
            dpsi_dx,
            x,
            grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True
        )[0]


        E = pinn.E # Energy eigenvalue

        

        #residual = (-0.5*d2psi_dx2 - E * psi)
        #if self.potential is not None:
        #    residual += self.potential(inputs) * psi
        #loss = torch.mean(residual**2)

        #norm = torch.trapz(psi[:,0]**2,x[:,0])

        #return self.weight * loss + self.weight_norm*(norm - 1)**2,d2psi_dx2
    
class PotentialHarmonicOscillator(Potential):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."

        x = inputs.clone().detach().requires_grad_(True)

        V = 0.5*x**2

        return V
    
def ansatzfactor_HO_sym(inputs: torch.Tensor, pinn: "Schrodinger1DTimeIndependentPINN"):
    assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
    x = inputs.clone().detach().requires_grad_(True)
    return torch.exp(-x**2/2)

def ansatzfactor_HO_asym(inputs: torch.Tensor, pinn: "Schrodinger1DTimeIndependentPINN"):
    assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
    x = inputs.clone().detach().requires_grad_(True)
    return torch.exp(-x**2/2)*x

def ansatzfactor_1Dbox(inputs: torch.Tensor,pinn: "Schrodinger1DTimeIndependentPINN"):
    assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
    x = inputs.clone().detach().requires_grad_(True)
    return (x+pinn.L)*(x-pinn.L)

# class LossBoundary(PhysicsLoss): #! Feil
#     def __call__(self, pinn: Schrodinger1DTimeIndependentPINN, inputs: torch.Tensor) -> torch.Tensor:
#         assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        
#         x = inputs.clone().detach().requires_grad_(True)

#         L = pinn.L

#         loss = x*(L-x)

#         return self.weight * loss