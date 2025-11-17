# src/core/neural_network.py
from __future__ import annotations
from typing import Generic
from torch import nn
from abc import ABC, abstractmethod
import torch


class FeedForwardNN(nn.Module):
    """Simple feed-forward neural network."""
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: int,
        width: int,
        activation_func: type[nn.Module] = nn.Tanh,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(activation_func())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(width, width))
            layers.append(activation_func())

        layers.append(nn.Linear(width, out_dim))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
