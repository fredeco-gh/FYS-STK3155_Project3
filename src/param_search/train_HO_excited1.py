#!/itf-fi-ml/home/heineeh/skole/FYS-STK3155/project3/.venv/bin/python

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))  # to import from src folder

import torch
from tise1d.tise1d import Loss_PDE, Loss_Orthogonality, PotentialHarmonicOscillator, ansatzfactor_HO_sym
from core.training import train_tise
from param_search.train_ground_state import load_ground_state_pinn
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

# System parameters
x_lim = (-5.0, 5.0)
N_samples = 256
E_excited_exact = 1.5  # Exact energy for first excited state of harmonic oscillator

# Generate data points
torch.manual_seed(124)
x = torch.rand(N_samples, 1, device=device) * (x_lim[1] - x_lim[0]) + x_lim[0]
x_test = torch.rand(N_samples, 1, device=device) * (x_lim[1] - x_lim[0]) + x_lim[0]


pinn1 = load_ground_state_pinn(device)


def main():
    # wandb.init() will pick up the sweep config defined on the web
    with wandb.init() as run:
        config = run.config

        step_method = {
            "Adam": torch.optim.Adam,
            "RMSProp": torch.optim.RMSprop,
        }[config.optimizer]

        activation_func = {
            "Tanh": torch.nn.Tanh,
            "ReLU": torch.nn.ReLU,
            "Sigmoid": torch.nn.Sigmoid,
        }[config.activation_function]

        # Train excited-state PINN with orthogonality to pinn1
        pinn = train_tise(
            x,
            loss_func=Loss_PDE(PotentialHarmonicOscillator()) + config.ortho_loss_weight*Loss_Orthogonality([pinn1]),
            x_lim=x_lim,
            lr=config.learning_rate,
            lr_energy=config.learning_rate_energy,
            step_method=step_method,
            hidden_layers=config.hidden_layers,
            width=config.network_width,
            E_init=0.5,
            activation_func=activation_func,
            ansatz_factor=ansatzfactor_HO_sym,
            lambd=config.weight_decay,
            device=device,
            verbose=False,
        )

        # --- Evaluation on test points ---
        loss_orthogonality = Loss_Orthogonality([pinn1])(pinn, x_test).detach().cpu().item()
        loss_pde = Loss_PDE(PotentialHarmonicOscillator())(pinn, x_test).detach().cpu().item()
        loss_energy = torch.abs((pinn.E.detach() - E_excited_exact) / E_excited_exact).detach().cpu().item()

        val_loss = loss_pde + 100 * loss_orthogonality

        run.log(
            {
                "validation_loss": val_loss,
                "Loss_PDE": loss_pde,
                "Loss_Orthogonality": loss_orthogonality,
                "Loss_Energy": loss_energy,
            }
        )

if __name__ == "__main__":
    main()
