from typing import Callable
from dataclasses import dataclass
import time
import numpy as np
import torch
from torch import nn, Tensor
import gymnasium as gym

from .utils import Ticker, Logger, Timer, Checkpointer


@dataclass
class RecurrentPPOConfig:
    total_steps: int = 1_000_000  # Total training environment steps
    rollout_steps: int = 32       # Number of vectorised steps per rollout
    num_envs: int = 32            # Number of parallel environments
    lr: float = 3e-4              # Optimiser learning rate
    decay_lr: bool = False        # Linear learning rate decay
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda parameter
    num_epochs: int = 10          # PPO epochs per update
    num_minibatches: int = 1      # PPO minibatch updates per epoch
    ppo_clip: float = 0.15        # PPO clipping epsilon
    value_loss_weight = 1.0       # Weight of value loss
    entropy_beta: float = 0.01    # Entropy regularisation coeffient
    advantage_norm: bool = True   # Normalise advantages if true
    grad_norm_clip: float = 0.5   # Global gradient norm clip
    network_hidden_dim: int = 64  # Hidden dim for default MLP
    gru_hidden_dim: int = 16      # GRU hidden dim
    cuda: bool = False            # Use GPU if available
    seed: int | None = 42         # RNG seed
    checkpoint: bool = False      # Enable model checkpointing
    save_interval: float = 600    # Checkpoint interval (seconds)
    verbose: bool = True          # Verbose logging

class GRUCore(nn.GRU):
    """GRU for RL with per-timestep hidden state resets."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)

    def forward(
        self, 
        x: Tensor,               # (T, B, input_dim)
        hx: Tensor | None,       # (1, B, H) | None
        dones: Tensor | None     # (T, B)    | None
    ) -> tuple[Tensor, Tensor]:  # (T, B, H), (1, B, H)
        # Initialise hidden state and dones if not provided
        seq_length, batch_size = x.shape[:2]
        hx = torch.zeros(
            1, batch_size, self.hidden_size, dtype=x.dtype, device=x.device
        ) if hx is None else hx
        dones = torch.zeros(
            seq_length, batch_size, dtype=torch.bool, device=x.device
        ) if dones is None else dones
        
        # Sequential GRU update loop
        outputs = []
        for t in range(seq_length):
            # Reset hidden state for start of new episodes
            hx[:, dones[t]] = 0.0

            # Step GRU for all environments at this timestep
            out, hx = super().forward(x[t:t+1], hx)
            outputs.append(out)

        # Concatenate outputs over time dimension
        gru_out = torch.concatenate(outputs, dim=0)
        return gru_out, hx

class RecurrentActorCritic(nn.Module):
    def __init__(
        self, 
        observation_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64, 
        gru_hidden_dim: int = 64
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru = GRUCore(hidden_dim, gru_hidden_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic_head = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        x: Tensor,                       # (T, B, *observation_shape)
        hx: Tensor | None,               # (1, B, H) | None
        dones: Tensor | None             # (T, B)    | None
    ) -> tuple[Tensor, Tensor, Tensor]:  # (T, B, A), (T, B), (1, B, H)
        x = self.base(x)
        x, hx = self.gru(x, hx, dones)
        logits = self.actor_head(x)
        values = self.critic_head(x).squeeze(-1)
        return logits, values, hx
    
def orthogonal_init(model: nn.Module, gain: float = 1.0):
    """Orthogonal weight and zero bias initialisation scheme."""
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
            self.network = RecurrentActorCritic(
                np.prod(self.envs.single_observation_space.shape),
                self.envs.single_action_space.n,
                hidden_dim=cfg.network_hidden_dim,
                gru_hidden_dim=cfg.gru_hidden_dim
            ).to(self.device)

        # Initialise network params with best practices for PPO
        orthogonal_init(self.network, gain=np.sqrt(2.0))
        self.network.actor_head[-1].weight.data.mul_(0.01)

        # Initialise Adam optimiser with larger epsilon
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=cfg.lr, eps=1e-5
        )

        # Linear learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
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

    def rollout(self) -> list[list[np.ndarray]]:
        """Collect a single rollout of experience across all vectorised envs."""
        experience = []

        # Observations, hx and done from initial reset or end of last rollout
        observations = self.current_observations
        hx = self.current_hx
        prev_dones = self.prev_dones
        
        for step_idx in range(self.cfg.rollout_steps):

            # Network forward pass, obtain actions, log probs and values
            observations_tensor = torch.as_tensor(observations[None, ...], dtype=torch.float32, device=self.device)
            prev_dones_tensor = torch.as_tensor(prev_dones[None, ...], dtype=torch.bool, device=self.device)
            with torch.inference_mode():
                logits, values, new_hx = self.network(observations_tensor, hx, prev_dones_tensor)
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            actions_tensor = dist.sample()
            log_probs = dist.log_prob(actions_tensor)
            
            # Vectorised environment step
            next_observations, rewards, terminations, truncations, infos = \
                self.envs.step(actions_tensor.cpu().numpy())
            
            # Handle next states in automatically reset environments
            final_observations = next_observations.copy()
            if "final_obs" in infos.keys():
                for obs_idx, obs in enumerate(infos["final_obs"]):
                    if obs is not None: final_observations[obs_idx] = obs

            # Get next values
            final_observations_tensor = torch.as_tensor(final_observations[None, ...], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                _, next_values, _ = self.network(
                    final_observations_tensor, 
                    new_hx,
                    dones=None
                )

            # Store transition in experience buffer
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
            
            # Log rewards and done info
            if self.ticker is not None:
                dones = np.logical_or(terminations, truncations)
                self.ticker.tick(rewards, dones)

            # Update observations and previous dones for next step
            observations = next_observations
            hx = new_hx
            prev_dones = np.logical_or(terminations, truncations)
            self.current_step += self.cfg.num_envs

        # Store last observations for start of next rollout
        self.current_observations = observations
        self.current_hx = hx
        self.prev_dones = prev_dones
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

        # Calculate GAE advantages and returns
        advantages = self.calculate_advantage(rewards, terminations, truncations, values, next_values)
        returns = values + advantages

        # Optional advantage normalisation
        if self.cfg.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Merge step and environment dims of tensors
        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        log_probs, actions, advantages, returns, values = \
            map(flatten, [log_probs, actions, advantages, returns, values])

        # Generate random indices, shape=(num_epochs, num_minibatches, minibatch_size)
        batch_size = self.cfg.rollout_steps * self.cfg.num_envs
        minibatch_size = batch_size // self.cfg.num_minibatches
        perms = np.stack([np.random.permutation(batch_size) for _ in range(self.cfg.num_epochs)])
        indices = perms.reshape(self.cfg.num_epochs, self.cfg.num_minibatches, minibatch_size)

        # PPO update loop: multiple epochs and minibatches
        for b_indices in indices:
            for mb_indices in b_indices:
                # Full forward pass with current network parameters
                new_logits, new_values, _ = self.network(observations, hx, prev_dones)
                
                # Flatten and slice minibatch indices
                new_logits, new_values = map(flatten, [new_logits, new_values])
                new_logits, new_values = new_logits[mb_indices], new_values[mb_indices]

                # Compute PPO clipped policy loss
                dist = torch.distributions.Categorical(logits=new_logits)
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
        self.lr_scheduler.step()

    def train(self) -> None:
        """Train Recurrent PPO agent."""
        if self.cfg.verbose: print("Training recurrent PPO agent")
        
        # Vectorised reset, get initial observations, set initial dones and hx
        self.current_observations, _ = self.envs.reset(seed=self.cfg.seed)
        self.prev_dones = np.zeros(self.cfg.num_envs, dtype=bool)
        self.current_hx = torch.zeros(1, self.cfg.num_envs, self.cfg.gru_hidden_dim, device=self.device)
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
