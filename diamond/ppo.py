from typing import Type
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch import nn
import gymnasium as gym

from .utils import Logger, Timer, Checkpointer


@dataclass
class PPOConfig:
    total_steps: int = 80_000
    rollout_steps: int = 64
    num_envs: int = 16
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 4
    num_minibatches: int = 8
    ppo_clip: float = 0.2
    value_loss_weight = 1.0
    entropy_beta: float = 0.01
    advantage_norm: bool = True
    grad_norm_clip: float = 0.5
    network_hidden_dim: int = 64
    network_activation_fn: Type[torch.nn.Module] = nn.Tanh
    device: torch.device = torch.device("cpu")
    checkpoint: bool = False
    checkpoint_save_interval_s: float = 600
    verbose: bool = True

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
        *,
        cfg: PPOConfig = PPOConfig(),
        custom_network: nn.Module | None = None
    ):
        # Create vectorised environments
        self.envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(cfg.num_envs)], 
            copy=True,
            autoreset_mode="SameStep"
        )

        # Create network and optimiser
        if custom_network is not None:
            self.network = custom_network.to(cfg.device)
        else:
            self.network = ActorCriticNetwork(
                np.prod(self.envs.single_observation_space.shape),
                self.envs.single_action_space.n,
                hidden_dim=cfg.network_hidden_dim,
                activation_fn=cfg.network_activation_fn
            ).to(cfg.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.learning_rate
        )

        # Create utility class instances
        self.logger = Logger(cfg.total_steps, cfg.num_envs, cfg.rollout_steps)
        self.timer = Timer()
        self.checkpointer = Checkpointer(folder="models", run_name="test")

        self.device = cfg.device
        self.cfg = cfg
        
    def select_action(self, observations: np.ndarray) -> np.ndarray:
        """NumPy action selection interface."""
        # Forward pass with policy network
        observations_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        with torch.inference_mode():
            logits = self.network.actor(observations_tensor)

        # Boltzmann action selection
        actions = torch.distributions.Categorical(logits=logits).sample()
        return actions.cpu().numpy()

    def rollout(self) -> list[list[np.ndarray]]:
        experience = []

        # Observations from initial reset or end of last rollout
        observations = self.current_observations

        # Perform rollout, storing transitions in experience buffer
        for step_idx in range(self.cfg.rollout_steps):
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

            # Append transition to experience buffer
            experience.append([observations, final_observations, actions, 
                                rewards, terminations, truncations])
            
            # Log transition with logger
            if self.logger is not None:
                self.logger.log(rewards, terminations, truncations)

            # Update observations
            observations = next_observations

        # Store last observations for start of next rollout
        self.current_observations = observations
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
        advantages = torch.zeros_like(rewards, device=self.cfg.device)
        advantage = 0.0
        for idx in reversed(range(self.cfg.rollout_steps)):
            non_termination, non_truncation = 1.0 - terminations[idx], 1.0 - truncations[idx]
            delta = rewards[idx] + self.cfg.gamma * next_values[idx] * non_termination - values[idx]
            advantages[idx] = advantage = delta + self.cfg.gamma * self.cfg.gae_lambda * non_termination * non_truncation * advantage
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

        # Log probs and values before any updates
        with torch.inference_mode():
            logits = self.network.actor(observations)
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            values = self.network.critic(observations).squeeze(-1)
            next_values = self.network.critic(next_observations).squeeze(-1)

        # Calculate advantages and returns
        advantages = self.calculate_advantage(
            rewards, terminations, truncations, values, next_values
        )
        returns = values + advantages

        if self.cfg.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Merge step and environment dims for each tensor
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        observations, log_probs, actions, advantages, returns, values = \
            map(flatten, [observations, log_probs, actions, advantages, returns, values])

        # Generate batch/minibatch indices
        batch_size = self.cfg.rollout_steps * self.cfg.num_envs
        minibatch_size = batch_size // self.cfg.num_minibatches
        perms = np.stack([np.random.permutation(batch_size) for _ in range(self.cfg.num_epochs)])
        indices = perms.reshape(self.cfg.num_epochs, self.cfg.num_minibatches, minibatch_size)

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
                    torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
                loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                # Value loss
                loss_value = 0.5 * torch.nn.functional.mse_loss(new_values.squeeze(1), returns[mb_indices])

                # Entropy loss
                entropy = dist.entropy().mean()

                # Combine and weight losses
                loss = (
                    loss_policy +
                    self.cfg.value_loss_weight * loss_value +
                    -self.cfg.entropy_beta * entropy
                )

                # Network update and global grad norm clip
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_norm_clip)
                self.optimizer.step()

    def train(self) -> None:
        """Train PPO agent."""
        # Initial reset
        self.current_observations, _ = self.envs.reset()
        self._last_checkpoint_time = time.time()

        # Main training loop
        total_rollouts = self.cfg.total_steps // (self.cfg.rollout_steps * self.cfg.num_envs)
        for rollout_idx in range(total_rollouts):
            # Perform rollout to gather experience
            experience = self.rollout()

            # Learn from gathered experience
            self.learn(experience)

            # Checkpointing
            if self.cfg.checkpoint:
                if time.time() - self._last_checkpoint_time >= self.cfg.checkpoint_save_interval_s:
                    self.checkpointer.save(self.logger.current_step, self.network, self.optimizer)
                    self._last_checkpoint_time = time.time()

        # Save final trained model
        if self.cfg.checkpoint:
            self.checkpointer.save(self.logger.current_step, self.network, self.optimizer)

        self.envs.close()
    