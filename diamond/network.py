"""
Lightweight PyTorch neural network components.

Includes:
- MLP: configurable multilayer perceptron.
"""

from typing import Iterable
import torch
from torch import nn


class MLP(nn.Module):
    """
    Multilayer Perceptron with optional layer normalisation.

    Parameters
    ------
    input_size: int
        Number of input features.
    hidden_sizes: Iterable[int]
        Sizes of hidden layers.
    output_size: int
        Number of output features.
    activation_fn: str
        Activation function to use ("relu", "tanh", "gelu").
    layer_norm: bool
        Whether to apply LayerNorm after each hidden layer.

    Example
    ------
    >>> model = MLP(10, (64, 64), 4, activation_fn="relu", layer_norm=True)
    >>> output = model(torch.randn(8, 10))
    """
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU
    }

    def __init__(
        self, 
        input_size: int,
        hidden_sizes: Iterable[int],
        output_size: int,
        activation_fn: str = "tanh",
        layer_norm: bool = False
    ):
        super().__init__()

        # Select activation function
        if activation_fn not in self.activations.keys():
            raise ValueError(f"Unsupported activation function: {activation_fn}")
        activation_cls = self.activations[activation_fn]

        # Collect layer sizes
        sizes = (input_size, *hidden_sizes, output_size)

        # Build the full MLP
        layers = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))

            # Add layer norm
            if layer_norm and out_dim != output_size:
                layers.append(nn.LayerNorm(out_dim))

            # Add activation fn
            if out_dim != output_size:
                layers.append(activation_cls())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    