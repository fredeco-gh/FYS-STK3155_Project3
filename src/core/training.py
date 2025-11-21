from __future__ import annotations
import torch
from torch import nn
from core.interfaces import AnsatzFactor,Potential
from core.neural_network import FeedForwardNN
from schrodinger_box.time_independent_1d import LossBoundary, NormLoss, Schrodinger1DTimeIndependentPINN, LossTISE1D, ansatzfactor_1Dbox
from torch.utils.data import DataLoader

def train_tise_example(
    L: float = 1.0,
    n_epochs: int = 5000,
    N_samples: int = 256,
    E: float = 0.5,
    hidden_layers: int = 3,
    width: int = 64,
    lambd: float = 0.0,
    lr: float = 1e-3,
    batch_size: int = 16,
    track_loss: bool = True,
    ansatz_factor: "AnsatzFactor | None" = ansatzfactor_1Dbox,
    potential: "Potential | None" = None,
    step_method: str = "Adam", 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    x = torch.linspace(-L,L,N_samples)
    input_loader = DataLoader(x, batch_size=batch_size, shuffle=True)
    device = torch.device(device)

    #Build the model
    model = FeedForwardNN(
        in_dim=1,
        out_dim=1,
        hidden_layers=hidden_layers,
        width=width,
        activation_func=nn.Tanh,
    ).to(device)
    pinn = Schrodinger1DTimeIndependentPINN(model, ansatz_factor,L=L, E = E).to(device)

    # Define loss
    total_loss_fn = LossTISE1D(potential)# + NormLoss() + LossBoundary()

    # Define optimizer
    
    if step_method == "Adam": 
        optimizer = torch.optim.Adam(pinn.parameters(), lr=lr,weight_decay=lambd)
    elif step_method == "RMSProp": 
        optimizer = torch.optim.RMSprop(pinn.parameters(), lr=lr,weight_decay=lambd)
    else:
        raise ValueError("step_method must be either 'Adam' or 'RMSProp'")

    loss_vals = []

    # Train the model
    for epoch in range(1, n_epochs + 1):

        #x_batch = torch.rand(N_samples, 1, device=device) * 2*L - L
        #input_loader = DataLoader(x_batch, batch_size=batch_size, shuffle=True)

        epoch_loss = torch.Tensor([0.0]).to(device)
        for batch in input_loader: 
            optimizer.zero_grad()
            loss = total_loss_fn(pinn, batch.view(-1, 1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss
        if track_loss == True: 
            loss_vals.append(epoch_loss.detach().numpy())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs}. Loss={float(epoch_loss.detach()):.3e}...", end="\r")
            # print(f"Individual losses: {total_loss_fn.compute_individual_losses(pinn, x.view(-1,1))}")
    

    # After training, return model and energy

    if track_loss: 
        return pinn, loss_vals
    return pinn, None