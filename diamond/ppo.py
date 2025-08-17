import time
from dataclasses import dataclass
from math import sqrt
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space, Box, Discrete

from .utils import Ticker, Logger, Timer, Checkpointer


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000    # total training environment steps
    rollout_steps: int = 64         # number of vectorised steps per rollout
    num_envs: int = 16              # number of parallel environments
    lr: float = 3e-4                # Adam optimiser learning rate
    adam_eps: float = 1e-5          # Adam optimiser epsilon
    decay_lr: bool = False          # linear learning rate decay
    gamma: float = 0.99             # discount factor
    gae_lambda: float = 0.95        # GAE lambda parameter
    num_epochs: int = 4             # PPO epochs per update
    num_minibatches: int = 8        # PPO minibatch updates per epoch
    ppo_clip: float = 0.2           # PPO clipping epsilon
    value_loss_weight: float = 1.0  # weight of value loss
    entropy_beta: float = 0.01      # entropy regularisation coeffient
    advantage_norm: bool = True     # normalise advantages if true
    grad_norm_clip: float = 0.5     # global gradient norm clip
    network_hidden_dim: int = 64    # hidden dim for default MLP
    cuda: bool = False              # use GPU if available
    seed: int | None = 42           # RNG seed
    checkpoint: bool = False        # enable model checkpointing
    save_interval: float = 600      # checkpoint interval (seconds)
    verbose: bool = True            # verbose logging


class ActorCriticNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: Space, 
        action_space: Space, 
        hidden_dim: int
    ) -> None:
        super().__init__()
        assert isinstance(observation_space, Box), "Only Box obs spaces are supported."
        assert isinstance(action_space, Discrete), "Only Discrete action spaces are supported."

        self.base = nn.Sequential(
            nn.Linear(int(np.prod(observation_space.shape)),  hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, int(action_space.n))
        )
        self.actor_out_layer = self.actor_head[-1]

        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim,  hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_actions(self, observations: np.ndarray, device: torch.device) -> np.ndarray:
        """Returns NumPy actions given a batch of NumPy observations."""
        observations_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        with torch.inference_mode():
            x = self.base(observations_tensor)
            logits = self.actor_head(x)

        # Boltzmann action selection
        actions = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        return actions
    
    def get_values(self, observations: torch.Tensor) -> torch.Tensor:
        """Returns state values given a batch of observations."""
        with torch.inference_mode():
            x = self.base(observations)
            values = self.critic_head(x)
        return values.squeeze(-1)
    
    def get_logits_and_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits and state values from a batch of observations."""
        x = self.base(x)
        logits = self.actor_head(x)
        values = self.critic_head(x).squeeze(-1)
        return logits, values


def network_parameter_init_(network: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal weight and zero bias initialisation scheme with small variance output layer."""
    with torch.no_grad():
        for m in network.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if hasattr(network, "actor_out_layer"):
            nn.init.orthogonal_(network.actor_out_layer.weight, gain=0.01)  # type: ignore


class PPO:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        cfg: PPOConfig = PPOConfig(),
        custom_network: nn.Module | None = None
    ) -> None:
        self.device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")

        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

        self.envs = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(cfg.num_envs)], 
            copy=True,
            autoreset_mode="Disabled"
        )

        if custom_network is not None:
            self.network = custom_network.to(self.device)
        else:
            self.network = ActorCriticNetwork(
                self.envs.single_observation_space,
                self.envs.single_action_space,
                hidden_dim=cfg.network_hidden_dim
            ).to(self.device)

            network_parameter_init_(self.network, gain=sqrt(2.0))

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.05 if cfg.decay_lr else 1.0,
            total_iters=cfg.total_steps // (cfg.num_envs * cfg.rollout_steps)
        )

        # Optional utilities for logging, timing and checkpointing
        self.logger = Logger()
        self.timer = Timer()
        self.checkpointer = Checkpointer(folder="models", run_name="default")
        self.ticker = Ticker(cfg.total_steps, cfg.num_envs, cfg.rollout_steps, verbose=cfg.verbose)
        
        self.current_step = 0
        self.cfg = cfg

    def rollout(self) -> list[list[np.ndarray]]:
        """Collect a single rollout of experience across all vectorised envs."""
        experience = []

        # Observations from initial reset or end of last rollout
        observations = self.current_observations

        for step_idx in range(self.cfg.rollout_steps):
            actions = self.network.get_actions(observations, device=self.device)  # type: ignore

            next_observations, rewards, terminations, truncations, infos = self.envs.step(actions)
            
            experience.append([
                observations, 
                next_observations, 
                actions, 
                rewards, 
                terminations, 
                truncations
            ])
            
            dones = np.logical_or(terminations, truncations)
            new_observations, infos = (
                self.envs.reset(options={"reset_mask": dones})
                if np.any(dones) else (next_observations, infos)
            )
            observations = new_observations

            if self.ticker is not None:
                self.ticker.tick(rewards, dones)

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
        """Calculate advantage with Generalised Advantage Estimation."""
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
        observations, next_observations, actions, rewards, terminations, truncations = zip(*experience)
        observations = torch.as_tensor(np.asarray(observations), dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(np.asarray(next_observations), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.asarray(terminations), dtype=torch.float32, device=self.device)
        truncations = torch.as_tensor(np.asarray(truncations), dtype=torch.float32, device=self.device)

        # Log probs and values before any updates
        with torch.inference_mode():
            logits, values = self.network.get_logits_and_values(observations)  # type: ignore
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            next_values = self.network.get_values(next_observations)  # type: ignore

        advantages = self.calculate_advantage(rewards, terminations, truncations, values, next_values)
        returns = values + advantages
        if self.cfg.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # Merge step and environment dims of each tensor
        observations, log_probs, actions, advantages, returns, values = [
            x.reshape(-1, *x.shape[2:])
            for x in (observations, log_probs, actions, advantages, returns, values)
        ]

        # Generate random indices, shape: (num_epochs, num_minibatches, minibatch_size)
        batch_size = self.cfg.rollout_steps * self.cfg.num_envs
        minibatch_size = batch_size // self.cfg.num_minibatches
        perms = np.stack([np.random.permutation(batch_size) for _ in range(self.cfg.num_epochs)])
        indices = perms.reshape(self.cfg.num_epochs, self.cfg.num_minibatches, minibatch_size)

        # PPO update loop
        for b_indices in indices:
            for mb_indices in b_indices:
                # Forward pass with current network parameters
                new_logits, new_values = self.network.get_logits_and_values(observations[mb_indices])  # type: ignore

                # Compute PPO clipped policy loss
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(actions[mb_indices])
                ratio = (new_log_probs - log_probs[mb_indices]).exp()
                loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                loss_surrogate_clipped = -advantages[mb_indices] * \
                    torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
                loss_policy = torch.max(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                loss_value = 0.5 * torch.nn.functional.mse_loss(new_values, returns[mb_indices])

                entropy = dist.entropy().mean()

                loss = (
                    loss_policy +
                    self.cfg.value_loss_weight * loss_value +
                    -self.cfg.entropy_beta * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.grad_norm_clip)
                self.optimizer.step()

        self.lr_scheduler.step()

    def train(self) -> None:
        """Train PPO agent."""
        self.current_observations, _ = self.envs.reset(seed=self.cfg.seed)

        last_checkpoint_time = time.time()
        total_rollouts = self.cfg.total_steps // (self.cfg.rollout_steps * self.cfg.num_envs)

        # Main training loop
        for rollout_idx in range(total_rollouts):
            experience = self.rollout()
            self.learn(experience)

            # Optionally save model state at intervals
            env_steps = (rollout_idx + 1) * self.cfg.rollout_steps * self.cfg.num_envs
            if self.cfg.checkpoint:
                if time.time() - last_checkpoint_time >= self.cfg.save_interval:
                    self.checkpointer.save(env_steps, self.network, self.optimizer)
                    last_checkpoint_time = time.time()

        # Optionally save final model after training
        if self.cfg.checkpoint:
            self.checkpointer.save(env_steps, self.network, self.optimizer)

        self.envs.close()
