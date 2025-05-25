from typing import Callable
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch import nn
import gymnasium as gym

from .utils import Logger, Timer, Checkpointer


@dataclass
class PPOConfig:
    total_steps: int = 400_000    # Total training environment steps
    rollout_steps: int = 64       # Number of vectorised steps per rollout
    num_envs: int = 16            # Number of parallel environments
    learning_rate: float = 3e-4   # Optimiser learning rate
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda parameter
    num_epochs: int = 4           # PPO epochs per update
    num_minibatches: int = 8      # PPO minibatch updates per epoch
    ppo_clip: float = 0.2         # PPO clipping epsilon
    value_loss_weight = 1.0       # Weight of value loss
    entropy_beta: float = 0.01    # Entropy regularisation coeffient
    advantage_norm: bool = True   # Normalise advantages if true
    grad_norm_clip: float = 0.5   # Global gradient norm clip
    network_hidden_dim: int = 64  # Hidden dim for default MLP
    cuda: bool = False            # Use GPU if available
    seed: int | None = 42         # RNG seed
    checkpoint: bool = False      # Enable model checkpointing
    save_interval: float = 600    # Checkpoint interval (seconds)
    verbose: bool = True          # Verbose logging

class ActorCriticNetwork(nn.Module):
    """Two hidden layer MLP."""
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64
    ):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim,  hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor[-1].weight.data *= 0.01
        
        self.critic = nn.Sequential(
            nn.Linear(observation_dim,  hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x) -> tuple[torch.Tensor]:
        return self.actor(x), self.critic(x)

class PPO:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        cfg: PPOConfig = PPOConfig(),
        custom_network: nn.Module | None = None
    ):
        # Device selection
        self.device = torch.device(
            "cuda" if cfg.cuda and torch.cuda.is_available() else "cpu"
        )

        # RNG seeding
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        # Create vectorised environments
        self.envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(cfg.num_envs)], 
            copy=True,
            autoreset_mode="SameStep"
        )

        # Create network and optimiser
        if custom_network is not None:
            self.network = custom_network.to(self.device)
        else:
            self.network = ActorCriticNetwork(
                np.prod(self.envs.single_observation_space.shape),
                self.envs.single_action_space.n,
                hidden_dim=cfg.network_hidden_dim
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.learning_rate
        )

        # Utilities for logging, timing and checkpointing
        self.logger = Logger(cfg.total_steps, cfg.num_envs, cfg.rollout_steps)
        self.timer = Timer()
        self.checkpointer = Checkpointer(folder="models", run_name="test")

        self.cfg = cfg
        
    def select_action(self, observations: np.ndarray) -> np.ndarray:
        """Sample discrete actions from current policy."""
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
        """Collect a single rollout of experience across all vectorised envs."""
        experience = []

        # Observations from initial reset or end of last rollout
        observations = self.current_observations

        for step_idx in range(self.cfg.rollout_steps):
            actions = self.select_action(observations)

            # Vectorised environment step
            next_observations, rewards, terminations, truncations, infos = \
                self.envs.step(actions)
            
            # Handle next states in automatically reset environments
            final_observations = next_observations.copy()
            if "final_obs" in infos.keys():
                for obs_idx, obs in enumerate(infos["final_obs"]):
                    if obs is not None: final_observations[obs_idx] = obs

            # Store transition in experience buffer
            experience.append([observations, final_observations, actions, 
                                rewards, terminations, truncations])
            
            # Log rewards and done info
            if self.logger is not None:
                self.logger.log(rewards, terminations, truncations)

            # Update observations for next step
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
        """
        Calculate advantage with Generalised Advantage Estimation.
        
        The advantage at each step is the current TD error (δ),
        plus the discounted (γ), decayed (λ) sum of future TD errors.
        Accumulation stops at terminations or truncations, as further TD errors
        are from future episodes.
        """
        advantages = torch.zeros_like(rewards, device=self.device)
        advantage = 0.0
        # Iterate backwards in time from last timestep to first
        for t in reversed(range(self.cfg.rollout_steps)):
            non_termination = 1.0 - terminations[t]
            non_truncation = 1.0 - truncations[t]

            # One-step TD error, do not bootstrap if terminated
            delta = (
                rewards[t] 
                + self.cfg.gamma * next_values[t] * non_termination 
                - values[t]
            )

            # Recursively accumulate discounted, decayed TD errors
            advantages[t] = advantage = (
                delta 
                + self.cfg.gamma 
                * self.cfg.gae_lambda 
                * non_termination 
                * non_truncation 
                * advantage
            )
        return advantages

    def learn(self, experience: list[list[np.ndarray]]) -> None:
        """Update policy and value networks using collected experience."""
        # Unpack experience and convert to PyTorch tensors
        observations, next_observations, actions, rewards, terminations, truncations = zip(*experience)
        observations = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(np.asarray(next_observations), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.asarray(terminations), dtype=torch.float32, device=self.device)
        truncations = torch.as_tensor(np.asarray(truncations), dtype=torch.float32, device=self.device)

        # Log probs and values before any updates
        with torch.inference_mode():
            logits = self.network.actor(observations)
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            values = self.network.critic(observations).squeeze(-1)
            next_values = self.network.critic(next_observations).squeeze(-1)

        # Calculate GAE advantages and returns
        advantages = self.calculate_advantage(rewards, terminations, truncations, values, next_values)
        returns = values + advantages

        # Optional advantage normalisation
        if self.cfg.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Merge step and environment dims of each tensor
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        observations, log_probs, actions, advantages, returns, values = \
            map(flatten, [observations, log_probs, actions, advantages, returns, values])

        # Generate random indices, shape=(num_epochs, num_minibatches, minibatch_size)
        batch_size = self.cfg.rollout_steps * self.cfg.num_envs
        minibatch_size = batch_size // self.cfg.num_minibatches
        perms = np.stack([np.random.permutation(batch_size) for _ in range(self.cfg.num_epochs)])
        indices = perms.reshape(self.cfg.num_epochs, self.cfg.num_minibatches, minibatch_size)

        # PPO update loop: multiple epochs and minibatches
        for b_indices in indices:
            for mb_indices in b_indices:
                # Forward pass with current network parameters
                new_logits, new_values = self.network(observations[mb_indices])

                # Compute PPO clipped policy loss
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions[mb_indices])
                ratio = (new_log_probs - log_probs[mb_indices]).exp()
                loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                loss_surrogate_clipped = -advantages[mb_indices] * \
                    torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
                loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                # MSE value loss
                loss_value = 0.5 * torch.nn.functional.mse_loss(new_values.squeeze(1), returns[mb_indices])

                # Entropy regularisation encourages exploration
                entropy = dist.entropy().mean()

                # Total loss
                loss = (
                    loss_policy +
                    self.cfg.value_loss_weight * loss_value +
                    -self.cfg.entropy_beta * entropy
                )

                # Update network parameters with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_norm_clip)
                self.optimizer.step()

    def train(self) -> None:
        """Train PPO agent."""
        # Vectorised reset, get initial observations
        self.current_observations, _ = self.envs.reset(seed=self.cfg.seed)
        last_checkpoint_time = time.time()

        # Compute number of rollouts to reach total steps
        total_rollouts = self.cfg.total_steps // (self.cfg.rollout_steps * self.cfg.num_envs)
        for rollout_idx in range(total_rollouts):
            # Gather experience with current policy
            experience = self.rollout()

            # Update policy and value networks from experience
            self.learn(experience)

            # Number of environment steps
            env_steps = (rollout_idx + 1) * self.cfg.rollout_steps * self.cfg.num_envs

            # Optionally save model state at intervals
            if self.cfg.checkpoint:
                if time.time() - last_checkpoint_time >= self.cfg.save_interval:
                    self.checkpointer.save(env_steps, self.network, self.optimizer)
                    last_checkpoint_time = time.time()

        # Optionally save final model after training
        if self.cfg.checkpoint:
            self.checkpointer.save(env_steps, self.network, self.optimizer)

        self.envs.close()
