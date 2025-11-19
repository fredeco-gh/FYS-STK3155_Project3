import torch
from torch import nn
from core.neural_network import FeedForwardNN
from schrodinger_box.time_independent_1d import Schrodinger1DTimeIndependentPINN, LossTISE1D
from torch.utils.data import DataLoader

def train_tise_example(
    L: float = 1.0,
    n_epochs: int = 5000,
    N_samples: int = 256,
    hidden_layers: int = 3,
    width: int = 64,
    lambd: float = 0.0,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    step_method: str = "Adam"
):
    x = torch.linspace(-L,L,256)
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
    pinn = Schrodinger1DTimeIndependentPINN(model, L=L, E_init=5.0).to(device)

    # Define loss
    total_loss_fn = LossTISE1D()

    # Define optimizer
    
    if step_method == "Adam": 
        optimizer = torch.optim.Adam(pinn.parameters(), lr=lr,weight_decay=lambd)
    elif step_method == "RMSProp": 
        optimizer = torch.optim.RMSprop(pinn.parameters(), lr=lr,weight_decay=lambd)

    # Train the model
    for epoch in range(1, n_epochs + 1):

        for batch in input_loader: 
            optimizer.zero_grad()
            loss = total_loss_fn(pinn, batch)
            loss.backward()
            optimizer.step()

    # After training, return model and energy
    return pinn