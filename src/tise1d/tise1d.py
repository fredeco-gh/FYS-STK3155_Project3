from __future__ import annotations
from core.interfaces import PhysicsLoss, PhysicsInformedNN, Potential, AnsatzFactor
import torch
import torch.nn as nn

class PINN(PhysicsInformedNN):
    """ Defining class for physics informed neural network (PINN)"""
    def __init__(self, model: nn.Module, ansatz_factor: AnsatzFactor | None, x_lim: tuple[float, float], E_init: float = 0.5):
        super().__init__(model,ansatz_factor)
        self.x_lim = x_lim
        self.E = nn.Parameter(torch.tensor(E_init, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        raw = self.model(inputs)    # result of neural network
        psi = raw*self.ansatz_factor(inputs,self)   # multiply result of neural network with an extra ansatz factor implementing boundary conditions
        return psi

class Loss_PDE(PhysicsLoss):
    """ Class defining the cost/loss of the differential equation"""
    def __init__(self, potential: "Potential | None" = None) -> None:
        super().__init__()
        self.potential = potential

    def __call__(self, pinn: "PINN", inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."

        # enable gradients w.r.t x
        x = inputs.clone().detach().requires_grad_(True)

        psi = pinn(x)  # (N,1)

        # first derivative
        dpsi_dx = torch.autograd.grad(
            outputs=psi,
            inputs=x,
            grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0]

        # second derivative
        d2psi_dx2 = torch.autograd.grad(
            outputs=dpsi_dx,
            inputs=x,
            grad_outputs=torch.ones_like(dpsi_dx),
            create_graph=True
        )[0]

        E = pinn.E  # trainable energy parameter

        # Hamiltonian applied to psi
        H_psi = -0.5 * d2psi_dx2
        if self.potential is not None:
            Vx = self.potential(x)   # (N,1)
            H_psi = H_psi + Vx * psi

        # PDE residual
        residual = H_psi - E*psi   

        loss =  torch.mean(residual**2)/torch.mean(psi**2) #divide by torch.mean(psi**2) to normalize the cost

        return self.weight * loss


class Loss_Orthogonality(PhysicsLoss):
    ''' Class defining the cost/loss for the orthogonality between different quantum states'''
    def __init__(self, reference_states: list["PINN"]) -> None:
        super().__init__()
        self.reference_states = reference_states
    
    def __call__(self, pinn: "PINN", inputs: torch.Tensor) -> torch.Tensor:
        # Use the device of the current network
        device = next(pinn.parameters()).device

        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        x = inputs.to(device)



        # current eigenfunction (n-th)
        psi_n = pinn(x)[:, 0]  # (N,)

        total = torch.Tensor([0.0]).to(x.device)
        for ref in self.reference_states:
            ref = ref.to(device)
            # detach ref to avoid backprop through it
            with torch.no_grad():
                psi_m = ref(x)[:, 0]  # (N,)

            # inner product ≈ L * mean(psi_n * psi_m)

            overlap_est = torch.mean(psi_n * psi_m)/torch.sqrt((torch.mean(psi_n**2)*torch.mean(psi_m**2)))
            total = total + overlap_est**2

        return self.weight * total.sum()

class PotentialHarmonicOscillator(Potential):
    ''' Potential energy for quantum harmonic oscillator'''
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        x = inputs.clone().detach().requires_grad_(True)

        V = 0.5*x**2

        return V

def ansatzfactor_box(inputs: torch.Tensor, pinn: "PINN"):
    ''' Ansatz factor implementing boundary conditions of 1D box [x_start,x_end]'''
    assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
    x = inputs

    x_start, x_end = pinn.x_lim
    
    return (x - x_start) * (x - x_end)

def ansatzfactor_HO_sym(inputs: torch.Tensor, pinn: "PINN"):
    ''' Ansatz factor implementing boundary conditions of quantum harmonic oscillator'''
    assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
    x = inputs
    return torch.exp(-x**2/2)