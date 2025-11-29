# %%
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))  # to import from src folder

import numpy as np
import torch
from core.grid_search import grid_search

# %% 

# === Example grid search usage ===
if __name__ == "__main__":
    from core.training import train_tise
    from tise1d.tise1d import Loss_PDE, PotentialHarmonicOscillator, Loss_Orthogonality, ansatzfactor_HO_sym
    from param_search.A_train_ground_state import load_ground_state_pinn
    from utils import generate_input_data
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    device ="cuda" if torch.cuda.is_available() else "cpu"
    
    x_lim = (-5.0, 5.0)
    x_test = generate_input_data(2048, x_lim, device)
    pinn1 = load_ground_state_pinn(device)

    def train_func(x: torch.Tensor, ortho_loss_weight: float, **kwargs):
        return train_tise(
            x,
            **kwargs,
            loss_func=Loss_PDE(PotentialHarmonicOscillator())+ortho_loss_weight*Loss_Orthogonality([pinn1]),
        )

    lr_values = np.logspace(np.log10(4e-5),np.log10(1e-2), 30)

    result = grid_search(
        train_func,
        metrics=[Loss_PDE(PotentialHarmonicOscillator()), Loss_Orthogonality([pinn1])],
        test_points=x_test,
        constant_parameters={
            "x_lim": x_lim,
            "ansatz_factor": ansatzfactor_HO_sym,
            "n_epochs": 3000,
            "E_init": 0.5,
            "device": device,
            "verbose": False,

            # Guess values
            "hidden_layers": 3,
            "width": 64,
            "ortho_loss_weight": 100,
            "lr_energy": 2e-2,
        },
        sweep_parameters={
            "lr": lr_values,
        },
        generate_data_func=lambda: generate_input_data(256, x_lim, device),
        n_repeats=10,
        seed=124
    )

    # %%

    # np.save("grid_search_tise.npy", result)
    # np.load("grid_search_tise.npy")

    # %%
    loss_pde_data = result[:, 0]
    loss_ortho_data = result[:, 1]

    fig,ax1 = plt.subplots()

    ax1.plot(lr_values, loss_pde_data,"-o",label="Loss PDE", c="C0")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Learning rate")
    ax1.set_ylabel("PDE loss")
    ax1.tick_params(axis="y",labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.plot(lr_values,loss_ortho_data,"-o",label="Loss Orthogonality", c="C1")
    ax2.set_ylabel("Orthogonality loss")
    ax2.tick_params(axis="y", labelcolor="C1")


    plt.legend()
    plt.show()

    # %%

    # Plotting the results
    # plots = [
    #     ("PDE Loss", loss_pde_data),
    #     ("Orthogonality Loss", loss_ortho_data),
    # ]

    # fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 5))
    # for ax, (title, data) in zip(axes, plots):
    #     pc = ax.pcolormesh(
    #         lr_values, lr_energy_values,
    #         data, 
    #         norm=LogNorm(),
    #         shading='auto'
    #     )
    #     fig.colorbar(pc, ax=ax, label=title)
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     ax.set_xlabel('Learning Rate Model')
    #     ax.set_ylabel('Learning Rate Energy')
    #     ax.set_title(title)
    # plt.tight_layout()
    # plt.show()
# %%
