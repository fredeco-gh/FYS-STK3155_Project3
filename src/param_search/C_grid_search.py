# %%
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))  # to import from src folder

import numpy as np
import torch
from core.grid_search import grid_search_parallel
from core.training import train_tise
from tise1d.tise1d import Loss_PDE, PotentialHarmonicOscillator, Loss_Orthogonality, ansatzfactor_HO_sym
from param_search.A_train_ground_state import load_ground_state_pinn
from utils import generate_input_data
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# device ="cuda" if torch.cuda.is_available() else "cpu"
x_lim = (-5.0, 5.0)
pinn1 = load_ground_state_pinn("cpu")


torch.manual_seed(124)
x_test = generate_input_data(2048, x_lim, "cpu")


def train_func(x: torch.Tensor, ortho_loss_weight: float, **kwargs):
    return train_tise(
        x,
        **kwargs,
        loss_func=Loss_PDE(PotentialHarmonicOscillator())+ortho_loss_weight*Loss_Orthogonality([pinn1]),
    )
def generate_data_func():
    return generate_input_data(256, x_lim, "cpu").numpy()


def main():
    lr_energy_values = np.logspace(-3, 0, 30)
    result = grid_search_parallel(
        train_func,
        metrics=[Loss_PDE(PotentialHarmonicOscillator()), Loss_Orthogonality([pinn1])],
        test_points=x_test,
        constant_parameters={
            "x_lim": x_lim,
            "ansatz_factor": ansatzfactor_HO_sym,
            "n_epochs": 3000,
            "E_init": 0.5,
            # "device": device,
            "verbose": False,

            # Guess values
            "hidden_layers": 3,
            "width": 64,
            "ortho_loss_weight": 100,
            "lr": 5e-2,
        },
        sweep_parameters={
            "lr_energy": lr_energy_values,
        },
        generate_data_func=generate_data_func,
        n_repeats=10,
        seed=124,
        max_workers=16,
        devices=["cuda:0","cuda:1","cuda:2","cuda:3"]
    )

    path = pathlib.Path(__file__).parent / "results" / "lr_energy_grid_search.npy"
    np.save(path, result)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()