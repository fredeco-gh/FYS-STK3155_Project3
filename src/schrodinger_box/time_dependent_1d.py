from __future__ import annotations
from typing import Callable
from core.interfaces import PhysicsLoss, PhysicsInformedNN, Potential, AnsatzFactor
import torch
import torch.nn as nn


class Schrodinger1DTimeDependentPINN(PhysicsInformedNN):
    def __init__(self, model: nn.Module, ansatz_factor: AnsatzFactor,initial_condition: Callable, L: float = 1.0,T: float = 1.0):
        super().__init__(model,ansatz_factor)
        self.L = L
        self.T = T
        self.initial_condition = initial_condition

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 2, "Expected input shape (N,2): x,t."

        raw = self.model(inputs)
        raw_c = torch.complex(raw[:,0:1],raw[:,1:2])

        psi = raw_c*self.ansatz_factor(inputs[:,0:1],self)*inputs[:,1:2] + self.initial_condition(inputs[:,0:1])

             #raw = torch.complex(raw[0], raw[1])

        return psi
    


class LossTDSE1D(PhysicsLoss):
    def __init__(self, potential: "Potential" | None) -> None:
        super().__init__()
        self.potential = potential

    """
    Physics loss for the 1D time-dependent Schrödinger equation, in units where hbar/m = 1.
    """
    def __call__(self, pinn: "Schrodinger1DTimeDependentPINN", inputs: torch.Tensor,num_T: float) -> torch.Tensor:
        assert inputs.shape[1] == 2, "Expected input shape (N,2): x,t."

        inputs = inputs.clone().detach().requires_grad_(True)

        psi = pinn(inputs)

        psi_der = torch.autograd.grad(
            psi,
            inputs,
            grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0]

        dpsi_dx = psi_der[:,0:1]
        dpsi_dt = psi_der[:,1:2]

        d2psi_dx2 = torch.autograd.grad(
            dpsi_dx,
            inputs,
            grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True
        )[0][:,0:1]

        residual = (-0.5*d2psi_dx2 - torch.complex(torch.tensor([0.0]),torch.tensor([1.0]))*dpsi_dt)
        if self.potential is not None:
            residual += self.potential(inputs[:,0:1]) * psi
        loss = torch.mean(torch.abs(residual)**2)

               #avg_norm = torch.trapz(torch.abs(psi)**2,inputs[:,0:1],dim=0)/num_T

        return self.weight * loss       # + 1.0*(avg_norm - 1)**2

class LossBoundary(PhysicsLoss):
    def __call__(self, pinn: "Schrodinger1DTimeDependentPINN", inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        
        length = pinn.L
        end_points = torch.tensor([[-length],[length]],dtype=torch.float32).requires_grad_(True)

        psi = pinn(end_points)

        loss = psi[0,0]**2 + psi[1,0]**2

        return self.weight * loss


class PotentialHarmonicOscillator(Potential):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."

        x = inputs.clone().detach().requires_grad_(True)

        V = 0.5*x**2

        return V
    
def ansatzfactor_HO_sym(inputs: torch.Tensor, pinn: "Schrodinger1DTimeDependentPINN"):
    x = inputs.clone().detach().requires_grad_(True)
    return torch.exp(-x**2/2)

def ansatzfactor_HO_asym(inputs: torch.Tensor, pinn: "Schrodinger1DTimeDependentPINN"):
    x = inputs.clone().detach().requires_grad_(True)
    return torch.exp(-x**2/2)*x

def ansatzfactor_1Dbox(inputs: torch.Tensor,pinn: "Schrodinger1DTimeDependentPINN"):
    x = inputs.clone().detach().requires_grad_(True)
    return (x+pinn.L)*(x-pinn.L)

def ansatzfactor_nothing(inputs: torch.Tensor,pinn: "Schrodinger1DTimeDependentPINN"):
    x = inputs.clone().detach().requires_grad_(True)
    return torch.ones_like(x)