from typing import Type
from dataclasses import dataclass
import numpy as np
import torch
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

        # Build buffer
        obs_shape = self.envs.single_observation_space.shape
        action_shape = self.envs.single_action_space.shape
        allocate_memory = lambda shape, dtype: torch.zeros(
            config.rollout_steps, config.num_envs, *shape,
            dtype=dtype, device=config.device
        )
        self.buffer = {
            "observations": allocate_memory(obs_shape, torch.float32),
            "next_observations": allocate_memory(obs_shape, torch.float32),
            "actions": allocate_memory(action_shape, torch.int64),
            "rewards": allocate_memory((), torch.float32),
            "terminations": allocate_memory((), torch.bool),
            "truncations": allocate_memory((), torch.bool),
            "log_probs": allocate_memory(action_shape, torch.float32),
            "values": allocate_memory((), torch.float32)
        }

        # Create network and optimiser
        self.network = ActorCriticNetwork(
            np.prod(obs_shape),
            self.envs.single_action_space.n,
            hidden_dim=config.network_hidden_dim,
            activation_fn=config.network_activation_fn
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )

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
            # Network forward pass
            observations_tensor = torch.tensor(
                observations, dtype=torch.float32, device=self.device
            )
            logits, values = self.network(observations_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            # Environment step
            next_observations, rewards, terminations, truncations, infos = \
                self.envs.step(actions.cpu().numpy())
            
            # Handle next states with reset environments

            # Add transition to buffer
            self.buffer["observations"][step_idx] = observations_tensor
            self.buffer["next_observations"][step_idx] = torch.as_tensor(
                next_observations, dtype=torch.float32, device=self.device
            )
            self.buffer["actions"][step_idx] = actions
            self.buffer["rewards"][step_idx] = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            self.buffer["terminations"][step_idx] = torch.as_tensor(
                terminations, dtype=torch.bool, device=self.device
            )
            self.buffer["truncations"][step_idx] = torch.as_tensor(
                truncations, dtype=torch.bool, device=self.device
            )
            self.buffer["log_probs"][step_idx] = log_probs
            self.buffer["values"][step_idx] = values.squeeze(-1)

            



    def learn(self):
        pass

    def train(self):
        pass
    