from __future__ import annotations
from typing import Callable
import torch
from torch import nn
from core.interfaces import AnsatzFactor,Potential, PhysicsLoss
from core.neural_network import FeedForwardNN
from torch.utils.data import DataLoader
from tise1d import tise1d

def train_tise(
    x,
    loss_func: PhysicsLoss,
    ansatz_factor: "AnsatzFactor | None",
    x_lim: tuple[float,float] = (0.0,1.0),
    n_epochs: int = 3000,
    E_init: float = 0.5,
    hidden_layers: int = 3,
    width: int = 32,
    lambd: float = 0.0,
    lr: float = 1e-2,
    lr_energy: float | None = None,
    batch_size: int = 256,
    step_method: Callable[..., torch.optim.Optimizer] = torch.optim.Adam, 
    activation_func: type[nn.Module] = nn.Tanh,
    verbose: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    input_loader = DataLoader(x, batch_size=batch_size, shuffle=True)

    #Build the model
    model = FeedForwardNN(
        in_dim=1,
        out_dim=1,
        hidden_layers=hidden_layers,
        width=width,
        activation_func=activation_func,
    ).to(device)
    pinn = tise1d.PINN(model, ansatz_factor, x_lim=x_lim, E_init=E_init).to(device)

    # Define optimizer
    
    if lr_energy is None:
        lr_energy = lr
    optimizer = step_method([
        {"params": pinn.model.parameters(), "lr": lr, "weight_decay": lambd},
        {"params": [pinn.E], "lr": lr_energy},
        
    ])

    # Train the model
    for epoch in range(1, n_epochs + 1):
        epoch_loss = torch.Tensor([0.0]).to(device)
        for batch in input_loader: 
            optimizer.zero_grad()
            loss = loss_func(pinn, batch.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs}. Loss={float(epoch_loss.detach()):.3e}...", end="\r")
            # print(f"Individual losses: {total_loss_fn.compute_individual_losses(pinn, x.view(-1,1))}")

    # After training, return model
    return pinn