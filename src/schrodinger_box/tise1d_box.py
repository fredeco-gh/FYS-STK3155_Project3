from __future__ import annotations
from core.interfaces import PhysicsLoss, PhysicsInformedNN, Potential, AnsatzFactor
import torch
import torch.nn as nn

class PINN(PhysicsInformedNN):
    def __init__(self, model: nn.Module, ansatz_factor: AnsatzFactor | None, L: float = 1.0, E: float = 0.5):
        super().__init__(model,ansatz_factor)
        self.L = L
        self.E = nn.Parameter(torch.tensor(E,dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        raw = self.model(inputs)
        psi = raw*self.ansatz_factor(inputs,self)
        return psi

class Loss_PDE(PhysicsLoss):
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

        E = pinn.E  # trainable scalar parameter

        # Hamiltonian applied to psi
        H_psi = -0.5 * d2psi_dx2
        if self.potential is not None:
            Vx = self.potential(x)   # (N,1)
            H_psi = H_psi + Vx * psi

        # PDE residual
        residual = H_psi - E * psi

        loss = torch.mean(residual**2)
        return self.weight * loss


class Loss_Norm(PhysicsLoss):
    def __call__(self, pinn: "PINN", inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."

        psi = pinn(inputs)  # (N,1)
        prob = psi**2

        L = pinn.L

        norm_est = prob.mean() * L  # Monte Carlo integral
        
        # Norm should be 1.
        loss = (norm_est - 1.0)**2
        return self.weight * loss

class Loss_Orthogonality(PhysicsLoss):
    def __init__(self, reference_states: list["PINN"]) -> None:
        super().__init__()
        self.reference_states = reference_states
    
    def __call__(self, pinn: "PINN", inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
        x = inputs

        # current eigenfunction (n-th)
        psi_n = pinn(x)[:, 0]  # (N,)

        total = torch.Tensor([0.0]).to(x.device)
        for ref in self.reference_states:
            # detach ref to avoid backprop through it
            with torch.no_grad():
                psi_m = ref(x)[:, 0]  # (N,)

            # inner product ≈ L * mean(psi_n * psi_m)
            overlap_est = pinn.L * torch.mean(psi_n * psi_m)
            total = total + overlap_est**2

        return self.weight * total.sum()


def ansatzfactor(inputs: torch.Tensor,pinn: "PINN"):
    assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
    x = inputs
    return x * (pinn.L - x)

def ansatzfactor_n2(inputs: torch.Tensor,pinn: "PINN"):
    assert inputs.shape[1] == 1, "Expected input shape (N,1): x."
    x = inputs
    return x * (pinn.L - x) * (pinn.L/2 - x)
