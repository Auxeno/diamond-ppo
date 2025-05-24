from typing import Type
from dataclasses import dataclass
import numpy as np
import torch
from torch.distributions import Categorical
from torch import nn
from gymnasium.vector import SyncVectorEnv


@dataclass
class PPOConfig:
    num_envs: int = 16
    rollout_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    ppo_clip: float = 0.2
    num_epochs: int = 4
    num_minibatches: int = 8
    gae_lambda: float = 0.95
    network_hidden_dim: int = 64
    network_activation_fn: Type[torch.nn.Module] = torch.nn.Tanh
    device: torch.device = torch.device("cpu")

class ActorCriticNetwork(nn.Module):
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64, 
        activation_fn: Type[nn.Module] = nn.Tanh
    ):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim,  hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor[-1].weight.data *= 0.01
        
        self.critic = nn.Sequential(
            nn.Linear(observation_dim,  hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPO:
    def __init__(
        self,
        env_fn: callable,
        config: PPOConfig
    ):
        # Create vectorised environments
        self.envs = SyncVectorEnv(
            [env_fn for _ in range(config.num_envs)], 
            copy=False,
            autoreset_mode="SameStep"
        )

        # Create network and optimiser
        self.network = ActorCriticNetwork(
            np.prod(self.envs.single_observation_space.shape),
            self.envs.single_action_space.n,
            hidden_dim=config.network_hidden_dim,
            activation_fn=config.network_activation_fn
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )

        self.buffer = []
        self.logger = None
        self.checkpointer = None
        self.device = config.device

    def select_action(self, observations: np.ndarray) -> np.ndarray:
        """Convenient action selection interface for trained agent."""
        observations_tensor = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )
        # Forward pass with policy network
        with torch.no_grad():
            logits = self.network.actor(observations_tensor)

        # Boltzmann action selection
        actions = torch.distributions.Categorical(logits=logits).sample()
        return actions.cpu().numpy()

    def rollout(self) -> tuple:
        # Observations from initial reset or end of last rollout
        observations = self.current_observations

        # Perform rollout, storing transitions in buffer
        for step_idx in range(self.config.rollout_steps):
            # Forward pass to obtain actions
            observations_tensor = torch.as_tensor(
                observations, device=self.device
            )
            dist = Categorical(logits=self.network.actor(observations_tensor))
            actions = dist.sample().cpu().numpy()

            # Environment step
            next_observations, rewards, terminations, truncations, infos = \
                self.envs.step(actions)
            
            # Handle next states with reset environments

            # Add transition to buffer
            self.buffer.append([observations, next_observations, actions, 
                                rewards, terminations, truncations])
            
            # Log transition
            if self.logger is not None:
                self.logger.log(actions, rewards, terminations, truncations)
            





    def learn(self):
        pass

    def train(self):
        pass
    