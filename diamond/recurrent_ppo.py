from typing import Callable
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch import nn
import gymnasium as gym

from .utils import Logger, Timer, Checkpointer


class GRUCore(nn.GRU):
    """GRU for RL with per-timestep hidden state resets."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim, batch_first=True)

    def forward(
        self, 
        x: torch.Tensor,            # (B, T, input_dim)
        hx: torch.Tensor | None,    # (1, B, hidden_dim) | None
        dones: torch.Tensor | None  # (B, T)             | None
    ):
        # Initialise hidden state and dones if not provided
        batch_size, seq_length = x.shape[:2]
        hx = torch.zeros(
            1, batch_size, self.gru.hidden_size, dtype=x.dtype, device=x.device
        ) if hx is None else hx
        dones = torch.zeros(
            batch_size, seq_length, dtype=torch.bool, device=x.device
        ) if dones is None else dones
        
        # Sequential GRU update loop
        outputs = []
        for t in range(seq_length):
            # Reset hidden state for start of new episodes
            hx[:, dones[:, t], :] = 0.0

            # Step GRU for all environments at this timestep
            out, hx = self.gru(x[:, t:t+1, :], hx)
            outputs.append(out)

        # Concatenate outputs over time dimension
        gru_out = torch.cat(outputs, dim=1)
        return gru_out, hx

class RecurrentActorCritic(nn.Module):
    """Simple recurrent actor-critic network."""
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64, 
        rnn_hidden_dim: int = 128
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = GRUCore(hidden_dim, rnn_hidden_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor_head[-1].weight.data *= 0.01
        self.critic_head = nn.Sequential(
            nn.Linear(rnn_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,            # (B, T, *observation_shape)
        hx: torch.Tensor | None,    # (1, B, rnn_hidden_dim) | None
        dones: torch.Tensor | None  # (B, T)                 | None
    ):
        x = self.base(x)
        x, hx = self.gru(x, hx, dones)
        logits = self.actor_head(x)
        values = self.critic_head(x).squeeze(-1)
        return logits, values, hx

