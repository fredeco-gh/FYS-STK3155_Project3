from core.interfaces import PhysicsLoss, PhysicsInformedNN
import torch
import torch.nn as nn


class Schrodinger1DTimeIndependentPINN(PhysicsInformedNN):
    def __init__(self, model: nn.Module, L: float = 1.0, E_init: float = 5.0):
        super().__init__(model)
        self.L = L

        # trainable eigenvalue parameter
        self.energy = nn.Parameter(torch.tensor(E_init, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        x = inputs

        raw = self.model(x)

        psi = x*(self.L-x)*raw
        return psi

class LossTISE1D(PhysicsLoss):
    """
    Physics loss for the 1D time-independent Schrödinger equation, in units where hbar/2m = 1.
    Potential V(x) is assumed to be 0 everywhere.
    """
    def __call__(self, pinn: Schrodinger1DTimeIndependentPINN, inputs: torch.Tensor) -> torch.Tensor:
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


        E = pinn.energy # Energy eigenvalue


        residual = -d2psi_dx2 - E * psi
        loss = torch.mean(residual**2)

        return self.weight * loss
