"""
Lightweight PyTorch neural network components.

Includes:
    MLP: Configurable multilayer perceptron.
    CNN: Nature CNN convolutional torso.
    ActorCritic: Simple actor-critic network with shared base.
"""
from typing import Iterable, Type
import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    """Configurable multilayer perceptron."""
    def __init__(
        self, 
        layer_sizes: Iterable[int],
        activation_fn: nn.Module = nn.Tanh,
    ):
        super().__init__()
        layers: list[nn.Module]= []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != layer_sizes[-1]:
                layers.append(activation_fn())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
class CNN:
    """Nature CNN convolutional torso for (H, W, C) observations."""
    def __init__(
        self,
        in_channels: int,
        activation_fn: Type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), 
            activation_fn(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            activation_fn(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), 
            activation_fn(),
            nn.Flatten()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.network(x)
    
class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes: Iterable[int],
        activation_fn: Type[nn.Module] = nn.Tanh
    ):
        super().__init__()
        pixel_obs = len(observation_space.shape) == 3

        # Shared torso
        if pixel_obs:
            w, h, c = observation_space.shape
            self.torso = CNN(in_channels=c, activation_fn=nn.ReLU)
            # Dummy forward pass to determine output size
            dummy_input = torch.zeros(1, h, w, c, dtype=torch.float32)
            with torch.no_grad():
                out = self.torso(dummy_input)
            torso_out_dim = out.shape[-1]
        else:
            self.torso = nn.Identity()
            torso_out_dim = np.prod(observation_space.shape)

        # Shared MLP
        self.mlp = MLP([torso_out_dim, *hidden_sizes], activation_fn)

        # Separate heads
        self.actor_head = nn.Linear(hidden_sizes[-1], action_space.n)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x: torch.Tensor):
        x = self.torso(x)
        x = self.mlp(x)
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        return logits, value
