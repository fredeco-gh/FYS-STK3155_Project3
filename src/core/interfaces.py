# src/core/interfaces.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields
from typing import Callable, TypeVar, Generic, Any
import torch
import copy



from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.neural_network import FeedForwardNN


PINN = TypeVar("PINN", bound="PhysicsInformedNN")
class PhysicsLoss(Generic[PINN], ABC):
    """Base class for physics-informed loss components. For example, differential equation residuals."""
    def __init__(self) -> None:
        super().__init__()
        self.weight: float = 1.0
        self.weight_norm = 0
        self.weight_BD = 1.0
    
    @abstractmethod
    def __call__(self, pinn: PINN, inputs: torch.Tensor) -> torch.Tensor:
        pass

    def __add__(self, other: PhysicsLoss) -> CompositePhysicsLoss:
        return CompositePhysicsLoss(self, other)
    
    def __mul__(self, scalar: float) -> PhysicsLoss:
        result = copy.copy(self)
        result.weight *= scalar
        return result
    
    def __rmul__(self, scalar: float) -> PhysicsLoss:
        return self.__mul__(scalar)

class Potential(Generic[PINN], ABC):
    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

class CompositePhysicsLoss(PhysicsLoss):
    """Composite loss that aggregates multiple PhysicsLoss components."""
    def __init__(self, *losses: PhysicsLoss):
        self.losses = losses

    def __call__(self, pinn: "PhysicsInformedNN", inputs: torch.Tensor) -> torch.Tensor:
        # Sum all losses
        losses = [loss(pinn, inputs) for loss in self.losses]
        return torch.stack(losses).sum()
    
    def __add__(self, other: PhysicsLoss) -> CompositePhysicsLoss:
        return CompositePhysicsLoss(*self.losses, other)

AnsatzFactor = Callable[[torch.Tensor, PINN], torch.Tensor]

class PhysicsInformedNN(ABC, torch.nn.Module):
    """Base class for physics-informed neural networks."""
    def __init__(self, model: torch.nn.Module,ansatz_factor: AnsatzFactor):
        super().__init__()
        self.model = model
        self.ansatz_factor = ansatz_factor

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the trial function output given the inputs."""
        pass
