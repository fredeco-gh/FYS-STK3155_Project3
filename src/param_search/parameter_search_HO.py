import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent)) # to import from src folder

import wandb
import pickle
import torch
from tise1d.tise1d import Loss_PDE, Loss_Orthogonality, PotentialHarmonicOscillator, ansatzfactor_HO_sym
from core.training import train_tise
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define system parameters
x_lim = (-3.0, 3.0)
N_samples = 256
E_excited_exact = 1.5  # Exact energy for first excited state of harmonic oscillator

# Define x values
torch.manual_seed(124)
x = torch.rand(N_samples, 1, device=device)*(x_lim[1]-x_lim[0])+x_lim[0]
x_test = torch.rand(N_samples, 1, device=device)*(x_lim[1]-x_lim[0])+x_lim[0]

# Train ground state PINN
pinn1 = train_tise(
    x,
    loss_func = Loss_PDE(PotentialHarmonicOscillator()),
    x_lim=x_lim,
    batch_size = 256,
    lr = 5e-2,
    lr_energy=1e-1,
    E_init = 0.1,
    ansatz_factor=ansatzfactor_HO_sym,
    device=device,
    verbose=False
)

# # Evaluate ground state PINN
# eval_loss1 = Loss_PDE(PotentialHarmonicOscillator())(pinn1, x_test)
# print(f"Ground state PINN evaluation loss: {eval_loss1:.3e}")


# ==== SWEEP ====
sweep_configuration = {
    "name": "sweep_ho_2",
    "method": "random",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"min": 1e-5, "max": 1e-1, "distribution": "log_uniform_values"},
        "learning_rate_energy": {"min": 1e-5, "max": 1e-1, "distribution": "log_uniform_values"},
        "optimizer": {"values": ["Adam", "RMSProp"]},
        "hidden_layers": {"values": [1, 2, 3, 4, 5]},
        "network_width": {"values": [32, 64, 128, 256, 512]},
        "ortho_loss_weight": {"min": 100, "max": 10000, "distribution": "log_uniform_values"},
        "activation_function": {"values": ["Tanh", "ReLU", "Sigmoid"]},
        "weight_decay": {"values": [0, 0, 0, 1e-7, 1e-3, 1]}, # {"min": 1e-10, "max": 1, "distribution": "log_uniform_values"}    
    },
}

def main():
    with wandb.init() as run:
        config = run.config

        step_method = {"Adam": torch.optim.Adam, "RMSProp": torch.optim.RMSprop}[config.optimizer]
        activation_func = {"Tanh": torch.nn.Tanh, "ReLU": torch.nn.ReLU, "Sigmoid": torch.nn.Sigmoid}[config.activation_function]

        pinn = train_tise(
            x,
            loss_func = Loss_PDE(PotentialHarmonicOscillator()) + config.ortho_loss_weight*Loss_Orthogonality([pinn1]),
            x_lim = x_lim,
            lr = config.learning_rate,
            lr_energy = config.learning_rate_energy,
            step_method=step_method,
            hidden_layers=config.hidden_layers,
            width=config.network_width,
            E_init = 0.5,
            activation_func=activation_func,
            ansatz_factor=ansatzfactor_HO_sym,
            lambd=config.weight_decay,
            device=device,
            verbose=False,
        )

        loss_orthogonality = Loss_Orthogonality([pinn1])(pinn, x_test).detach().cpu().item()
        loss_pde = Loss_PDE()(pinn, x_test).detach().cpu().item()
        loss_energy = torch.abs((pinn.E.detach() - 1.5)/1.5).detach().cpu().item()

        val_loss = loss_pde + 3000*loss_orthogonality
        
        run.log({"validation_loss": val_loss, "Loss_PDE": loss_pde, "Loss_Orthogonality": loss_orthogonality, "Loss_Energy": loss_energy})

# Initialize the sweep
sweep_id2 = wandb.sweep(sweep=sweep_configuration, project="sweep_ho_2")
# Run sweep
wandb.agent(sweep_id2, function=main)
