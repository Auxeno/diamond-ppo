from typing import Type
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import gymnasium as gym


@dataclass
class PPOConfig:
    total_steps = 80_000
    rollout_steps: int = 128
    num_envs: int = 16
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 4
    num_minibatches: int = 8
    ppo_clip: float = 0.2
    value_loss_weight = 0.5
    entropy_beta = 0.01
    advantage_norm: bool = True
    grad_norm_clip = 0.5
    network_hidden_dim: int = 64
    network_activation_fn: Type[torch.nn.Module] = nn.Tanh
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
        
    def forward(self, x) -> tuple[torch.Tensor]:
        return self.actor(x), self.critic(x)

class PPO:
    def __init__(
        self,
        env_fn: callable,
        config: PPOConfig = PPOConfig()
    ):
        # Create vectorised environments
        self.envs = gym.vector.SyncVectorEnv(
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

        self.rollout_count = 0
        self.logger = None
        self.checkpointer = None
        self.config = config
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

    def rollout(self) -> list[list[np.ndarray]]:
        experience = []

        # Observations from initial reset or end of last rollout
        observations = self.current_observations

        # Perform rollout, storing transitions in experience buffer
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
                    if obs is not None: final_observations[obs_idx] = obs

            # Add transition to experience buffer
            experience.append([observations, final_observations, actions, 
                                rewards, terminations, truncations])
            
            # Log transition
            if self.logger is not None:
                self.logger.log(actions, rewards, terminations, truncations)

        # Store last observations for start of next rollout
        self.current_observations = next_observations
        self.rollout_count +=1 

        return experience

    def calculate_advantage(
        self, 
        rewards: torch.Tensor, 
        terminations: torch.Tensor, 
        truncations: torch.Tensor, 
        values: torch.Tensor, 
        next_values: torch.Tensor
    ) -> torch.Tensor:
        """Calculate advantage with generalised advantage estimation."""
        advantages = torch.zeros_like(rewards, device=self.config.device)
        advantage = 0.0
        for idx in reversed(range(self.config.rollout_steps)):
            non_termination, non_truncation = 1.0 - terminations[idx], 1.0 - truncations[idx]
            delta = rewards[idx] + self.config.gamma * next_values[idx] * non_termination - values[idx]
            advantages[idx] = advantage = delta + self.config.gamma * self.config.gae_lambda * non_termination * non_truncation * advantage
        return advantages

    def learn(self, experience: list[list[np.ndarray]]) -> None:
        # Unpack experience
        observations, next_observations, actions, rewards, terminations, truncations = zip(*experience)
        observations = torch.as_tensor(np.array(observations), dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(np.array(next_observations), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.array(terminations), dtype=torch.float32, device=self.device)
        truncations = torch.as_tensor(np.array(truncations), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Log probs and values before any updates
            logits = self.network.actor(observations)
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            values = self.network.critic(observations).squeeze(-1)
            next_values = self.network.critic(next_observations).squeeze(-1)

        # Calculate advantages and returns
        advantages = self.calculate_advantage(
            rewards, terminations, truncations, values, next_values
        )
        returns = values + advantages

        # Normalise advantages
        if self.config.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Merge step and environment dims for each tensor
        flatten = lambda x: x.view(-1, *x.shape[2:])
        observations, log_probs, actions, advantages, returns, values = (
            map(flatten, [observations, log_probs, actions, advantages, returns, values])
        )

        # Generate indices
        batch_size = self.config.rollout_steps * self.config.num_envs
        minibatch_size = batch_size // self.config.num_minibatches
        perms = np.stack([np.random.permutation(batch_size) for _ in range(self.config.num_epochs)])
        indices = perms.reshape(self.config.num_epochs, self.config.num_minibatches, minibatch_size)

        # PPO learning steps
        for b_indices in indices:
            for mb_indices in b_indices:
                # Forward pass with current parameters
                new_logits, new_values = self.network(observations[mb_indices])

                # PPO policy loss
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions[mb_indices])
                ratio = (new_log_probs - log_probs[mb_indices]).exp()
                loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                loss_surrogate_clipped = -advantages[mb_indices] * \
                    torch.clamp(ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip)
                loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                # Value loss
                loss_value = torch.nn.functional.mse_loss(new_values.squeeze(1), returns[mb_indices])

                # Entropy loss
                entropy = dist.entropy().mean()

                # Combine and weight losses
                loss = (
                    loss_policy +
                    self.config.value_loss_weight * loss_value +
                    -self.config.entropy_beta * entropy
                )

                # Network update and global grad norm clip
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

    def train(self) -> None:
        """Train PPO agent."""
        # Initial reset
        self.current_observations, _ = self.envs.reset()

        # Main training loop
        total_rollouts = self.config.total_steps // (self.config.rollout_steps * self.config.num_envs)
        for rollout_idx in range(total_rollouts):
            # Perform rollout to gather experience
            experience = self.rollout()

            # Learn from gathered experience
            self.learn(experience)

        self.envs.close()
    