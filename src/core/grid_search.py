
import numpy as np
import torch
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from itertools import product
from core.interfaces import PINN


def enumerated_product(*args):
    indices = product(*(range(len(x)) for x in args))
    values = product(*args)
    
    yield from zip(indices, values)


def grid_search(
        train_func: Callable[..., PINN],
        metrics: list[Callable[[PINN, torch.Tensor], torch.Tensor|float]],
        test_points: torch.Tensor, # Points to evaluate the metrics on
        sweep_parameters: dict[str, list|NDArray], 
        constant_parameters: dict[str, Any] | None = None,
        n_repeats: int = 1,
        generate_data_func: Callable[[], torch.Tensor] | None = None,
        seed: int | None = None
    ) -> NDArray:

    if len(sweep_parameters) == 0:
        raise ValueError("No parameters to sweep over.")
    if len(metrics) == 0:
        raise ValueError("No metrics provided for evaluation.")
    if constant_parameters is None:
        constant_parameters = {}

    shape = (*[len(v) for v in sweep_parameters.values()], len(metrics))
    data = np.zeros(shape)
 
    n_combinations = np.prod([len(v) for v in sweep_parameters.values()])
    for i, (idx, value) in enumerate(enumerated_product(*sweep_parameters.values())):
        print(f"Running combination {i+1}/{n_combinations}", end='\r')
        current_sweep_params = {k: v for k, v in zip(sweep_parameters.keys(), value)}

        if seed is not None:
            torch.manual_seed(seed)

        for r in range(n_repeats):
            if generate_data_func is not None:
                # Generate new data for each repeat
                input_data = generate_data_func()
                # Input data has to be the first parameter of train_func
                model = train_func(input_data, **constant_parameters, **current_sweep_params)
            else:
                model = train_func(**constant_parameters, **current_sweep_params)
            
            for i, metric in enumerate(metrics):
                metric_value = metric(model, test_points)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.detach().cpu().item()
                data[*idx, i] += metric_value
        
        data[*idx] /= n_repeats

    return data