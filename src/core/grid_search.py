import numpy as np
import torch
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from itertools import product
from core.interfaces import PINN
from concurrent.futures import ProcessPoolExecutor, as_completed


def enumerated_product(*args):
    indices = product(*(range(len(x)) for x in args))
    values = product(*args)
    
    yield from zip(indices, values)

def run_single_config(
        train_func: Callable[..., "PINN"],
        metrics: list[Callable[["PINN", torch.Tensor], torch.Tensor | float]],
        test_points_np: np.ndarray,
        parameters: dict[str, Any],
        n_repeats: int,
        device: str,
        generate_data_func: Callable[[], np.ndarray] | None = None,
        seed: int | None = None
    ) -> list[float]:
        """
        Run a single configuration of hyperparameters multiple times and average the metrics.
        """
    
        # reconstruct tensors on the correct device
        test_points = torch.from_numpy(test_points_np).to(device)

        metric_sums = np.zeros(len(metrics), dtype=float)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


        for _ in range(n_repeats):
            if generate_data_func is not None:
                # Generate new data for each repeat
                input_data_np = generate_data_func()
                input_data = torch.from_numpy(input_data_np).to(device)
                # Input data has to be the first parameter of train_func
                model = train_func(input_data, **parameters)
            else:
                model = train_func(**parameters)
            
            for i, metric_func in enumerate(metrics):
                metric_value = metric_func(model, test_points)
                if isinstance(metric_value, torch.Tensor):
                    metric_value = metric_value.detach().cpu().item()
                metric_sums[i] += float(metric_value)
        
        return  (metric_sums / n_repeats).tolist()


def grid_search(
        train_func: Callable[..., PINN],
        metrics: list[Callable[[PINN, torch.Tensor], torch.Tensor|float]],
        test_points: torch.Tensor, # Points to evaluate the metrics on
        sweep_parameters: dict[str, list|NDArray], 
        constant_parameters: dict[str, Any] | None = None,
        n_repeats: int = 1,
        generate_data_func: Callable[[], NDArray[np.floating]] | None = None,
        seed: int | None = None
    ) -> NDArray:
    """
    Grid search over hyperparameters.
    """

    if len(sweep_parameters) == 0:
        raise ValueError("No parameters to sweep over.")
    if len(metrics) == 0:
        raise ValueError("No metrics provided for evaluation.")
    if constant_parameters is None:
        constant_parameters = {}

    shape = (*[len(v) for v in sweep_parameters.values()], len(metrics))
    data = np.zeros(shape)

    test_points_np = test_points.cpu().numpy()
 
    n_combinations = np.prod([len(v) for v in sweep_parameters.values()])
    for i, (idx, values) in enumerate(enumerated_product(*sweep_parameters.values())):
        print(f"Running combination {i+1}/{n_combinations}", end='\r')
        current_sweep_params = {k: v for k, v in zip(sweep_parameters.keys(), values)}

        data[*idx] = run_single_config(
            train_func,
            metrics,
            test_points_np,
            {**constant_parameters, **current_sweep_params},
            n_repeats,
            device=test_points.device.type,
            generate_data_func=generate_data_func,
            seed=seed
        )

    return data

def grid_search_parallel(
    train_func: Callable[..., PINN],
    metrics: list[Callable[[PINN, torch.Tensor], torch.Tensor|float]],
    test_points: torch.Tensor, # On cpu
    sweep_parameters: dict[str, list|NDArray], 
    constant_parameters: dict[str, Any] | None = None,
    n_repeats: int = 1,
    generate_data_func: Callable[[], NDArray[np.floating]] | None = None,
    max_workers: int | None = None,
    seed: int | None = None,
    devices: list[str] | None = None,
) -> NDArray[np.floating]:
    """
    Parallel grid search over hyperparameters, optionally using multiple GPUs.
    """
    if len(sweep_parameters) == 0:
        raise ValueError("No parameters to sweep over.")
    if len(metrics) == 0:
        raise ValueError("No metrics provided for evaluation.")
    if constant_parameters is None:
        constant_parameters = {}
    
    shape = (*[len(v) for v in sweep_parameters.values()], len(metrics))
    data = np.zeros(shape, dtype=float)

    if devices is None:
        devices = ["cpu"]
    
    test_points_np = test_points.cpu().numpy()

    n_combinations = np.prod([len(v) for v in sweep_parameters.values()])
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (idx, values) in enumerate(enumerated_product(*sweep_parameters.values())):
            device = devices[i % len(devices)]
            current_sweep_params = {k: v for k, v in zip(sweep_parameters.keys(), values)}
            fut = executor.submit(
                run_single_config,
                train_func,
                metrics,
                test_points_np,
                {**constant_parameters, **current_sweep_params},
                n_repeats,
                device,
                generate_data_func,
                seed
            )
            futures[fut] = idx
        
        for fut in as_completed(futures):
            idx = futures[fut]
            metric_vals = fut.result()
            data[idx] = metric_vals
            completed += 1
            print(f"Completed configuration {completed}/{n_combinations}", end='\r')

    return data