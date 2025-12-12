import torch
from IPython.core.magic import register_cell_magic
from IPython.core.getipython import get_ipython


# Generates the x input data from a uniform distribution
def generate_input_data(N_samples: int, x_lim: tuple[float, float], device: str) -> torch.Tensor:
    x = torch.rand(N_samples, 1, device=device) * (x_lim[1] - x_lim[0]) + x_lim[0]
    return x

@register_cell_magic
def skip_if(line, cell):
    """Jupyter cell magic to skip execution of a cell based on a condition."""
    global_scope = get_ipython().user_global_ns
    if eval(line, global_scope):
        return  # Skip execution if the condition is True
    get_ipython().run_cell(cell) # Execute the cell if the condition is False