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
    advantage_norm: bool = True
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
        self.rollout_count = 0
        self.logger = None
        self.checkpointer = None
        self.device = config.device

    def select_action(self, observations: np.ndarray) -> np.ndarray:
        """NumPy action selection interface."""
        observations_tensor = torch.as_tensor(
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
            # Action selection
            actions = self.select_action(observations)

            # Environment step
            next_observations, rewards, terminations, truncations, infos = \
                self.envs.step(actions)
            
            # Handle next states in automatically reset environments
            final_observations = next_observations.copy()
            if "final_obs" in infos.keys():
                for obs_idx, obs in enumerate(infos["final_obs"]):
                    final_observations[obs_idx] = obs

            # Add transition to buffer
            self.buffer.append([observations, final_observations, actions, 
                                rewards, terminations, truncations])
            
            # Log transition
            if self.logger is not None:
                self.logger.log(actions, rewards, terminations, truncations)

        # Store last observations for start of next rollout
        self.current_observations = next_observations
        self.rollout_count +=1 

    def calculate_advantage(self, rewards, terminations, truncations, values, next_values):
        """Calculate advantage with generalised advantage estimation."""
        advantages = torch.zeros_like(rewards, device=self.config.device)
        advantage = 0.0
        for idx in reversed(range(self.config.rollout_steps)):
            non_termination, non_truncation = 1.0 - terminations[idx], 1.0 - truncations[idx]
            delta = rewards[idx] + self.config.gamma * next_values[idx] * non_termination - values[idx]
            advantages[idx] = advantage = delta + self.config.gamma * self.config.gae_lambda * non_termination * non_truncation * advantage
        return advantages

    def learn(self):
        # Unpack experience from, then clear buffer
        observations, next_observations, actions, rewards, terminations, truncations = zip(*self.buffer)
        observations = torch.as_tensor(np.array(observations), dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(np.array(next_observations), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.array(terminations), dtype=torch.bool, device=self.device)
        truncations = torch.as_tensor(np.array(truncations), dtype=torch.bool, device=self.device)
        self.buffer = []

        with torch.no_grad():
            # Log probs and values before any updates
            logits = self.network.actor(observations)
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            values = self.network.critic(observations).squeeze(-1)
            next_values = self.network.critic(next_observations).squeeze(-1)

        # Calculate advantages
        advantages = self.calculate_advantage(
            rewards, terminations, truncations, values, next_values
        )

        # Normalise advantages
        if self.config.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def train(self):
        pass
    