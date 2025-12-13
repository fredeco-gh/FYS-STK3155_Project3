import torch


# Generates the x input data from a uniform distribution
def generate_input_data(N_samples: int, x_lim: tuple[float, float], device: str) -> torch.Tensor:
    x = torch.rand(N_samples, 1, device=device) * (x_lim[1] - x_lim[0]) + x_lim[0]
    return x
