from typing import Callable
from dataclasses import dataclass
import time
import numpy as np

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

class ActorCriticNetwork:
    pass

class PPO:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        cfg: PPOConfig = PPOConfig(),
        custom_network
    ):
        # Device selection
        self.device = None

        # RNG seeding
        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        # Create vectorised environments
        self.envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(cfg.num_envs)], 
            copy=True,
            autoreset_mode="SameStep"
        )

        # Create network and optimiser
        if custom_network is not None:
            self.network = custom_network
        else:
            self.network = ActorCriticNetwork()
        self.optimizer = None

        # Utilities for logging, timing and checkpointing
        self.logger = Logger(cfg.total_steps, cfg.num_envs, cfg.rollout_steps)
        self.timer = Timer()
        self.checkpointer = Checkpointer(folder="models", run_name="test")

        self.cfg = cfg
        
    def select_action(self, observations: np.ndarray) -> np.ndarray:
        """Sample discrete actions from current policy."""
       pass

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
        rewards, 
        terminations, 
        truncations,, 
        values, 
        next_values
    ):
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
        pass

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
