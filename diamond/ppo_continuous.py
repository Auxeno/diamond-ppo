from typing import Callable
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch import nn, Tensor
import gymnasium as gym
from gymnasium.spaces import Box

from .utils import Ticker, Logger, Timer, Checkpointer


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000  # Total training environment steps
    rollout_steps: int = 64       # Number of vectorised steps per rollout
    num_envs: int = 16            # Number of parallel environments
    learning_rate: float = 3e-4   # Optimiser learning rate
    decay_lr: bool = True         # Linear learning rate decay
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

class JointNormal(torch.distributions.Normal):
    def log_prob(self, value: Tensor) -> Tensor:
        """Return joint log-probability over all action dimensions."""
        return super().log_prob(value).sum(-1)

    def entropy(self) -> Tensor:
        """Return joint entropy over all action dimensions."""
        return super().entropy().sum(-1)

class ActorCriticNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: Box, 
        action_space: Box, 
        hidden_dim: int = 64
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(int(np.prod(observation_space.shape)),  hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor_mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, int(np.prod(action_space.shape)))
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1,int(np.prod(action_space.shape))))
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim,  hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def actor(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns action mean and log std."""
        x = self.base(x)
        mean = self.actor_mean_head(x)
        log_std = torch.broadcast_to(self.actor_log_std, mean.shape)
        return mean, log_std
    
    def critic(self, x: Tensor) -> Tensor:
        """Returns state value estimates given observations."""
        x = self.base(x)
        return self.critic_head(x).squeeze(-1)
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns action logits and state values from observations."""
        x = self.base(x)
        mean = self.actor_mean_head(x)
        log_std = torch.broadcast_to(self.actor_log_std, mean.shape)
        value = self.critic_head(x).squeeze(-1)
        return mean, log_std, value

def orthogonal_init(model: nn.Module, gain: float = 1.0):
    """Orthogonal weight and zero bias initialisation scheme."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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

        # Set up network
        if custom_network is not None:
            self.network = custom_network.to(self.device)
        else:
            self.network = ActorCriticNetwork(
                self.envs.single_observation_space,
                self.envs.single_action_space,
                hidden_dim=cfg.network_hidden_dim
            ).to(self.device)

        # Initialise network params with best practices for PPO
        orthogonal_init(self.network, gain=np.sqrt(2.0))

        # Initialise Adam optimiser with larger epsilon
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.learning_rate, eps=1e-5
        )

        # Linear learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.05 if cfg.decay_lr else 1.0,
            total_iters=cfg.total_steps // (cfg.num_envs * cfg.rollout_steps)
        )

        # Track current step
        self.current_step = 0

        # Utilities for logging, timing and checkpointing
        self.ticker = Ticker(cfg.total_steps, cfg.num_envs, cfg.rollout_steps,
                             verbose=cfg.verbose)
        self.logger = Logger()
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
            mean, log_std = self.network.actor(observations_tensor)

        # Boltzmann action selection
        actions = JointNormal(loc=mean, scale=log_std.exp()).sample()
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
            experience.append([
                observations, 
                final_observations, 
                actions, 
                rewards, 
                terminations, 
                truncations
            ])
            
            # Update rewards and done info in ticker
            if self.ticker is not None:
                dones = np.logical_or(terminations, truncations)
                self.ticker.tick(rewards, dones)

            # Update observations for next step and step counter
            observations = next_observations
            self.current_step += self.cfg.num_envs

        # Store last observations for start of next rollout
        self.current_observations = observations
        return experience

    def calculate_advantage(
        self, 
        rewards: Tensor, 
        terminations: Tensor, 
        truncations: Tensor, 
        values: Tensor, 
        next_values: Tensor
    ) -> Tensor:
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
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.asarray(terminations), dtype=torch.float32, device=self.device)
        truncations = torch.as_tensor(np.asarray(truncations), dtype=torch.float32, device=self.device)

        # Log probs and values before any updates
        with torch.inference_mode():
            means, log_stds, values = self.network(observations)
            log_probs = JointNormal(loc=means, scale=log_stds.exp()).log_prob(actions)
            next_values = self.network.critic(next_observations)

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
                new_means, new_log_stds, new_values = self.network(observations[mb_indices])

                # Compute PPO clipped policy loss
                dist = JointNormal(loc=new_means, scale=new_log_stds.exp())
                new_log_probs = dist.log_prob(actions[mb_indices])
                ratio = (new_log_probs - log_probs[mb_indices]).exp()
                loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                loss_surrogate_clipped = -advantages[mb_indices] * \
                    torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
                loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                # MSE value loss
                loss_value = 0.5 * torch.nn.functional.mse_loss(new_values, returns[mb_indices])

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

        # Step learning rate scheduler
        self.scheduler.step()

    def train(self) -> None:
        """Train PPO agent."""
        if self.cfg.verbose: print("Training PPO agent")

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
