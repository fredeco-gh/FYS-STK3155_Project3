import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))  # to import from src folder

import torch
from tise1d.tise1d import Loss_PDE, PotentialHarmonicOscillator, ansatzfactor_HO_sym
from core.training import train_tise
from utils import generate_input_data
from tise1d.analytic_ho import compare_analytic, energy_analytic


x_lim = (-5.0, 5.0)
save_path = pathlib.Path(__file__).parent / "models/ground_state_params.pt"

def train_ground_state_pinn(device):
    N_samples = 256

    torch.manual_seed(124)
    x = generate_input_data(N_samples, x_lim, device=device)
    

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
    return pinn1



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

    state_dict = torch.load(save_path, map_location=device)
    pinn1.load_state_dict(state_dict)
    pinn1.eval()
    return pinn1


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pinn1 = train_ground_state_pinn(device)

    # Save pinn1 parameters to file
    torch.save(pinn1.state_dict(), save_path)

    # Load pinn1 for evaluation
    # pinn1 = load_ground_state_pinn('cpu')

    # Evaluate
    torch.manual_seed(124)
    x_test = generate_input_data(1024, x_lim, device=device)
    loss = Loss_PDE(PotentialHarmonicOscillator())(pinn1, x_test).detach().cpu().item()
    
    compare_analytic(x_test, x_lim, pinn1, n=0)


    