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
class RecurrentPPOConfig:
    total_steps: int = 1_000_000    # total training environment steps
    rollout_steps: int = 32         # number of vectorised steps per rollout
    num_envs: int = 32              # number of parallel environments
    lr: float = 3e-4                # Adam optimiser learning rate
    adam_eps: float = 1e-5          # Adam optimiser epsilon
    decay_lr: bool = False          # linear learning rate decay
    gamma: float = 0.99             # discount factor
    gae_lambda: float = 0.95        # GAE lambda parameter
    num_epochs: int = 10            # PPO epochs per update
    num_minibatches: int = 1        # PPO minibatch updates per epoch
    ppo_clip: float = 0.15          # PPO clipping epsilon
    value_loss_weight: float = 1.0  # weight of value loss
    entropy_beta: float = 0.01      # entropy regularisation coeffient
    advantage_norm: bool = True     # normalise advantages if true
    grad_norm_clip: float = 0.5     # global gradient norm clip
    network_hidden_dim: int = 64    # hidden dim for default MLP
    gru_hidden_dim: int = 16        # GRU hidden dim
    cuda: bool = False              # use GPU if available
    seed: int | None = 42           # RNG seed
    checkpoint: bool = False        # enable model checkpointing
    save_interval: float = 600      # checkpoint interval (seconds)
    verbose: bool = True            # verbose logging


class GRUCore(nn.GRU):
    """GRU for RL with per-timestep hidden state resets."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__(input_dim, hidden_dim)

    def forward(
        self, 
        x: torch.Tensor,
        hx: torch.Tensor | None,
        dones: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GRU forward with per-timestep hidden resets.

        Parameters
        ----------
        x : (T, B, input_dim) torch.Tensor, dtype=float32
            Input sequence.
        hx : (1, B, H) torch.Tensor, dtype=float32 or None
            Initial hidden state, set to zeros if None.
        dones : (T, B) torch.Tensor, dtype=bool or None
            Episode done mask, if True resets hidden state.

        Returns
        -------
        gru_out : (T, B, H) torch.Tensor, dtype=float32
            GRU outputs per timestep.
        hx : (1, B, H) torch.Tensor, dtype=float32
            Final hidden state.

        Notes
        -----
        T: sequence length, B: batch size, H: GRU hidden size.
        """

        # Initialise hidden state and dones if not provided
        seq_length, batch_size = x.shape[:2]
        hx = torch.zeros(1, batch_size, self.hidden_size, dtype=x.dtype, device=x.device) if hx is None else hx
        dones = torch.zeros(seq_length, batch_size, dtype=torch.bool, device=x.device) if dones is None else dones
        
        outputs = []
        for t in range(seq_length):
            # Reset hidden state for start of new episodes
            hx[:, dones[t]] = 0.0

            out, hx = super().forward(x[t:t+1], hx)
            outputs.append(out)

        # Concatenate over time dimension
        gru_out = torch.concatenate(outputs, dim=0)
        return gru_out, hx


class RecurrentActorCritic(nn.Module):
    def __init__(
        self, 
        observation_space: Space, 
        action_space: Space, 
        hidden_dim: int = 64, 
        gru_hidden_dim: int = 64
    ) -> None:
        super().__init__()
        assert isinstance(observation_space, Box), "Only Box obs spaces are supported."
        assert isinstance(action_space, Discrete), "Only Discrete action spaces are supported."

        self.base = nn.Sequential(
            nn.Linear(int(np.prod(observation_space.shape)), hidden_dim),
            nn.Tanh(),
        )

        self.gru = GRUCore(hidden_dim, gru_hidden_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, int(action_space.n))
        )

        self.critic_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        hx: torch.Tensor | None,
        dones: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Actor-critic forward with GRU core.

        Parameters
        ----------
        x : (T, B, *observation_shape) torch.Tensor, dtype=float32
            Input observation sequence.
        hx : (1, B, H) torch.Tensor, dtype=float32 or None
            Initial GRU hidden state, set to zeros if None.
        dones : (T, B) torch.Tensor, dtype=bool or None
            Episode termination mask, True resets hidden state.

        Returns
        -------
        logits : (T, B, A) torch.Tensor, dtype=float32
            Action logits.
        values : (T, B) torch.Tensor, dtype=float32
            State-value estimates.
        hx : (1, B, H) torch.Tensor, dtype=float32
            Final GRU hidden state.

        Notes
        -----
        T: sequence length, B: batch size, H: GRU hidden size, A: number of actions.
        """
        x = self.base(x)
        x, hx = self.gru.forward(x, hx, dones)
        logits = self.actor_head(x)
        values = self.critic_head(x).squeeze(-1)
        return logits, values, hx
    

def orthogonal_init_(model: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal weight and zero bias initialisation scheme."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class RecurrentPPO:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        cfg: RecurrentPPOConfig = RecurrentPPOConfig(),
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
            self.network = RecurrentActorCritic(
                self.envs.single_observation_space,
                self.envs.single_action_space,
                hidden_dim=cfg.network_hidden_dim,
                gru_hidden_dim=cfg.gru_hidden_dim
            ).to(self.device)

            # Initialise network params with best practices for PPO
            orthogonal_init_(self.network, gain=sqrt(2.0))
            self.network.actor_head[-1].weight.data.mul_(0.01)  # type: ignore

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.05 if cfg.decay_lr else 1.0,
            total_iters=cfg.total_steps // (cfg.num_envs * cfg.rollout_steps)
        )

        # Utilities for logging, timing and checkpointing
        self.logger = Logger()
        self.timer = Timer()
        self.checkpointer = Checkpointer(folder="models", run_name="default")
        self.ticker = Ticker(cfg.total_steps, cfg.num_envs, cfg.rollout_steps, verbose=cfg.verbose)

        self.cfg = cfg

    def rollout(self) -> list[list[np.ndarray]]:
        """Collect a single rollout of experience across all vectorised envs."""
        experience = []

        # Observations, hx and done from initial reset or end of last rollout
        observations = self.current_observations
        hx = self.current_hx
        prev_dones = self.prev_dones
        
        for step_idx in range(self.cfg.rollout_steps):
            observations_tensor = torch.as_tensor(observations[None, ...], dtype=torch.float32, device=self.device)
            prev_dones_tensor = torch.as_tensor(prev_dones[None, ...], dtype=torch.bool, device=self.device)
            with torch.inference_mode():
                logits, values, new_hx = self.network(observations_tensor, hx, prev_dones_tensor)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            actions_tensor = dist.sample()
            log_probs = dist.log_prob(actions_tensor)
            
            next_observations, rewards, terminations, truncations, infos = self.envs.step(actions_tensor.cpu().numpy())

            # Get next values
            final_observations_tensor = torch.as_tensor(next_observations[None, ...], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                _, next_values, _ = self.network(
                    final_observations_tensor, 
                    new_hx,
                    dones=None
                )

            experience.append([
                observations_tensor.squeeze(0),
                actions_tensor,
                rewards,
                terminations,
                truncations,
                prev_dones_tensor.squeeze(0),
                log_probs,
                values.squeeze(0),
                next_values.squeeze(0),
                hx
            ])

            dones = np.logical_or(terminations, truncations)
            new_observations, infos = (
                self.envs.reset(options={"reset_mask": dones})
                if np.any(dones) else (next_observations, infos)
            )
            observations = new_observations
            hx = new_hx
            prev_dones = np.logical_or(terminations, truncations)

            if self.ticker is not None:
                self.ticker.tick(rewards, dones)

        # Store last observations for start of next rollout
        self.current_observations = observations
        self.current_hx = hx
        self.prev_dones = prev_dones
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

        # Unpack and convert experience
        observations, actions, rewards, terminations, truncations, prev_dones, log_probs, values, next_values, hx = zip(*experience)
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device)
        terminations = torch.as_tensor(np.asarray(terminations), dtype=torch.float32, device=self.device)
        truncations = torch.as_tensor(np.asarray(truncations), dtype=torch.float32, device=self.device)
        prev_dones = torch.stack(prev_dones)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        next_values = torch.stack(next_values)
        hx = hx[0].clone()

        advantages = self.calculate_advantage(rewards, terminations, truncations, values, next_values)
        returns = values + advantages
        if self.cfg.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Merge step and environment dims of each tensor
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        log_probs, actions, advantages, returns, values = [
            flatten(x)
            for x in (log_probs, actions, advantages, returns, values)
        ]

        # Generate random indices, shape: (num_epochs, num_minibatches, minibatch_size)
        batch_size = self.cfg.rollout_steps * self.cfg.num_envs
        minibatch_size = batch_size // self.cfg.num_minibatches
        perms = np.stack([np.random.permutation(batch_size) for _ in range(self.cfg.num_epochs)])
        indices = perms.reshape(self.cfg.num_epochs, self.cfg.num_minibatches, minibatch_size)

        # PPO update loop
        for b_indices in indices:
            for mb_indices in b_indices:
                # Full forward pass with current network parameters
                new_logits, new_values, _ = self.network(observations, hx, prev_dones)
                
                # Flatten and slice minibatch indices
                new_logits, new_values = [flatten(x) for x in [new_logits, new_values]]
                new_logits, new_values = new_logits[mb_indices], new_values[mb_indices]

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
        """Train recurrent PPO agent."""
        self.current_observations, _ = self.envs.reset(seed=self.cfg.seed)
        self.prev_dones = np.zeros(self.cfg.num_envs, dtype=bool)
        self.current_hx = torch.zeros(1, self.cfg.num_envs, self.cfg.gru_hidden_dim, device=self.device)

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
