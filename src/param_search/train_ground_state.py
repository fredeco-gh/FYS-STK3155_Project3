import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))  # to import from src folder

import torch
from tise1d.tise1d import Loss_PDE, PotentialHarmonicOscillator, ansatzfactor_HO_sym
from core.training import train_tise


x_lim = (-5.0, 5.0)
path = pathlib.Path(__file__).parent / "pinn1_ground_state.pt"

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    N_samples = 256

    torch.manual_seed(124)
    x = torch.rand(N_samples, 1, device=device) * (x_lim[1] - x_lim[0]) + x_lim[0]

    pinn1 = train_tise(
        x,
        loss_func=Loss_PDE(PotentialHarmonicOscillator()),
        x_lim=x_lim,
        batch_size=256,
        lr=5e-2,
        lr_energy=1e-1,
        E_init=0.1,
        ansatz_factor=ansatzfactor_HO_sym,
        device=device,
        verbose=True,
    )

    # Evaluate ground state PINN
    eval_loss1 = Loss_PDE(PotentialHarmonicOscillator())(pinn1, x)
    print(f"Ground state PINN evaluation loss: {eval_loss1:.3e}")


    # Save to current folder
    torch.save(pinn1.state_dict(), path)



# Function for loading later
def load_ground_state_pinn(device):
    from tise1d.tise1d import PINN as TisePINN
    from core.neural_network import FeedForwardNN
    import torch.nn as nn

    model = FeedForwardNN(
        in_dim=1,
        out_dim=1,
        hidden_layers=3,
        width=32,
        activation_func=nn.Tanh,
    ).to(device)
    pinn1 = TisePINN(model, ansatzfactor_HO_sym, x_lim=x_lim, E_init=0.1).to(device)

    state_dict = torch.load(path, map_location=device)
    pinn1.load_state_dict(state_dict)
    pinn1.eval()
    return pinn1