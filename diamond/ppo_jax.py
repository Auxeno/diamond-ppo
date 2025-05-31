from typing import Sequence, Any, Tuple, Union, Callable
import time
import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.struct import dataclass, field
from flax.linen.initializers import orthogonal, he_normal
import optax
from chex import Scalar, Array, ArrayTree, PRNGKey
from distrax import Categorical

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
    hidden_dims: Sequence[int] = (64, 64)  # Hidden dim for MLP
    cuda: bool = False            # Use GPU if available
    seed: int = 42         # RNG seed
    checkpoint: bool = False      # Enable model checkpointing
    save_interval: float = 600    # Checkpoint interval (seconds)
    verbose: bool = True          # Verbose logging

# --- Networks ---

class MLPTorso(nn.Module):
    """MLP torso network for vector observations."""
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:        
        for dim in self.hidden_dims:
            x = nn.Dense(dim, kernel_init=orthogonal(jnp.sqrt(2.0)))(x)
            x = nn.relu(x)
        return x

class SimpleCNNTorso(nn.Module):
    """Simple CNN torso network for pixel observations."""
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            kernel_init=he_normal()
        )(x)
        x = nn.relu(x)
        x = x.reshape(*x.shape[:-3], -1)
        x = MLPTorso(self.hidden_dims)(x)
        return x

class ActorCriticNetwork(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int]
    pixel_obs: bool
    
    def setup(self) -> None:
        if self.pixel_obs:
            self.torso = SimpleCNNTorso(self.hidden_dims)
        else:
            self.torso = MLPTorso(self.hidden_dims)
        self.actor_head = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))
        self.critic_head = nn.Dense(1, kernel_init=orthogonal(1.0))
    
    def actor_fn(self, x: Array) -> Array:
        """Output action logits."""
        x = self.torso(x)
        x = self.actor_head(x)
        return x
        
    def critic_fn(self, x: Array) -> Array:
        """Output value for state."""
        x = self.torso(x)
        x = self.critic_head(x).squeeze(-1)
        return x

    def __call__(self, x: Array) -> Tuple[Array, Array]:
        """Combined forward pass of actor and critic."""
        x = self.torso(x)
        logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        return logits, value
    
# --- Pytrees ---

@dataclass
class Transition:
    """Transition for a single step in vectorised environments."""
    observations: Array = field(pytree_node=True)
    next_observations: Array = field(pytree_node=True)
    actions: Array = field(pytree_node=True)
    rewards: Array = field(pytree_node=True)
    terminations: Array = field(pytree_node=True)
    truncations: Array = field(pytree_node=True)
    log_probs: Array = field(pytree_node=True)
    values: Array = field(pytree_node=True)

class AgentState(TrainState):
    """TrainState superclass for separate calling of actor and critic."""
    actor_fn: Callable = field(pytree_node=False)
    critic_fn: Callable = field(pytree_node=False)

# --- PPO Agent ---
    
class PPO:
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        *,
        cfg: PPOConfig = PPOConfig(),
        custom_network: nn.Module | None = None
    ):
        # Device selection
        self.device = None

        # RNG seeding
        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        # Create vectorised environments
        self.envs = SyncVectorEnv(
            [env_fn for _ in range(cfg.num_envs)], 
            copy=True,
            autoreset_mode="SameStep"
        )

        # Initialise agent state
        self.agent_state = self.create_agent_state(cfg, self.envs, custom_network)

        # Utilities for logging, timing and checkpointing
        self.logger = Logger(cfg.total_steps, cfg.num_envs, cfg.rollout_steps)
        self.timer = Timer()
        self.checkpointer = None

        self.cfg = cfg

    def create_agent_state(
        self, 
        cfg: PPOConfig, 
        envs: SyncVectorEnv,
        custom_network: nn.Module | None
    ) -> AgentState:
        """Initialise PPO agent state."""
        # Build network
        if custom_network is not None:
            network = custom_network
        else:
            network = ActorCriticNetwork(
                action_dim=self.envs.single_action_space.n,
                hidden_dims=cfg.hidden_dims,
                pixel_obs=len(envs.single_observation_space.shape) == 3
            )
        
        # Initialise network parameters
        key = jax.random.PRNGKey(cfg.seed)
        sample_obs = envs.observation_space.sample()
        params = network.init(key, sample_obs)
        
        # Configure optimiser with optional gradient clipping
        if cfg.grad_norm_clip is not None:
            optimizer = optax.chain(
                optax.clip_by_global_norm(cfg.grad_norm_clip),
                optax.adam(learning_rate=cfg.learning_rate, eps=1e-8)
            )
        else:
            optimizer = optax.adam(learning_rate=learning_rate, eps=1e-8)

        return AgentState.create(
            apply_fn=network.apply,
            actor_fn=lambda params, x: network.apply(params, x, method=network.actor_fn),
            critic_fn=lambda params, x: network.apply(params, x, method=network.critic_fn),
            params=params,
            tx=optimizer
        )
        
    @staticmethod
    def select_action(
        key: PRNGKey, 
        agent_state: AgentState, 
        observations: Array, 
    ) -> tuple[Array, Array, Array]:
        """Select action with policy network."""
        # Select actions as well as calculate log probs and values
        logits, values = agent_state.apply_fn(agent_state.params, observations)
        dist = Categorical(logits=logits)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    def rollout(self, key: PRNGKey) -> ArrayTree:
        """Collect a single rollout of experience across all vectorised envs."""
        observations = self.current_observations

        experience = []

        # Observations from initial reset or end of last rollout
        observations = self.current_observations

        for step_idx in range(self.cfg.rollout_steps):
            key, key_action = self.split(key)

            # Select action using policy
            actions, log_probs, values = self.select_action(key_action, self.agent_state, observations)

            # Vectorised environment step
            next_observations, rewards, terminations, truncations, infos = \
                self.envs.step(np.array(actions))
            
            # Handle next states in automatically reset environments
            final_observations = next_observations.copy()
            if "final_obs" in infos.keys():
                for obs_idx, obs in enumerate(infos["final_obs"]):
                    if obs is not None: final_observations[obs_idx] = obs
            
            # Log rewards and done info
            if self.logger is not None:
                self.logger.log(rewards, terminations, truncations)

            # Add transition to transition list
            experience.append(Transition(
                observations=observations,
                next_observations=final_observations,
                actions=actions,
                rewards=rewards,
                terminations=terminations,
                truncations=truncations,
                log_probs=log_probs,
                values=values
            ))

            # Update observations for next step
            observations = next_observations

        # Store last observations for start of next rollout
        self.current_observations = observations
        return experience

    def calculate_advantage(
        self, 
        rewards, 
        terminations, 
        truncations,
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

    @staticmethod
    def learn(key: PRNGKey, agent_state: AgentState, batch: Transition, cfg: PPOConfig) -> AgentState:
        """Perform PPO learning update."""
        def calculate_gae(agent_state: AgentState, batch: Transition) -> Tuple:
            """Compute advantage and returns using GAE."""

            def gae_step(advantage, transition) -> Tuple[Array, ArrayTree]:
                """Scannable GAE step."""
                reward, termination, truncation, value, next_value = transition
                non_termination, non_truncation = 1.0 - termination, 1.0 - truncation
                delta = reward + cfg.gamma * next_value * non_termination - value
                advantage = delta + cfg.gamma * cfg.gae_lambda * non_termination * non_truncation * advantage
                return advantage, advantage

            # Compute values for next observations
            next_values = agent_state.critic_fn(agent_state.params, batch.next_observations)

            # Initialise GAE scan parameters
            initial_carry = jnp.zeros(cfg.num_envs)
            transitions = (batch.rewards, batch.terminations, batch.truncations, batch.values, next_values)

            # Compute advantages via reversed scan
            _, advantages = jax.lax.scan(
                gae_step,
                initial_carry,
                transitions,
                reverse=True
            )

            # Calculate returns
            returns = advantages + batch.values

            return advantages, returns

        def minibatch_update(agent_state: AgentState, mb_indices: Array) -> Tuple[AgentState, Any]:
            """Scannable minibatch gradient descent update."""

            def ppo_loss(params: ArrayTree) -> Scalar:
                """Differentiable PPO loss function."""

                # Forward pass
                logits, values = agent_state.apply_fn(params, batch.observations[mb_indices])

                # Policy loss
                distribution = Categorical(logits=logits)
                log_probs = distribution.log_prob(batch.actions[mb_indices])
                ratio = jnp.exp(log_probs - batch.log_probs[mb_indices])
                loss_surrogate_unclipped = -advantages[mb_indices] * ratio
                loss_surrogate_clipped = -advantages[mb_indices] * \
                    jnp.clip(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip)
                loss_policy = jnp.maximum(loss_surrogate_unclipped, loss_surrogate_clipped).mean()

                # Value loss
                loss_value = ((values - returns[mb_indices]) ** 2).mean()

                # Entropy bonus
                loss_entropy = distribution.entropy().mean()

                # Combine losses
                loss = (
                    loss_policy +
                    cfg.value_loss_weight * loss_value +
                   -cfg.entropy_beta * loss_entropy
                )
                return loss

            # Calculate PPO loss and gradients
            loss, grads = jax.value_and_grad(ppo_loss)(agent_state.params)

            # Update model parameters
            agent_state = agent_state.apply_gradients(grads=grads)

            return agent_state, loss

        # Compute advantages using GAE
        advantages, returns = calculate_gae(agent_state, batch)

        # Normalise advantages if enabled            
        advantages = jnp.where(
            cfg.advantage_norm,
            (advantages - advantages.mean()) / (advantages.std() + 1e-8),
            advantages
        )

        # Flatten batch data
        batch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
        advantages, returns = jax.tree.map(lambda x: x.flatten(), (advantages, returns))

        # Create shuffled minibatch indices
        batch_size = cfg.rollout_steps * cfg.num_envs
        indices = jnp.tile(jnp.arange(batch_size), (cfg.num_epochs, 1))
        indices = jax.vmap(jax.random.permutation)(jax.random.split(key, cfg.num_epochs), indices)
        indices = indices.reshape(cfg.num_epochs * cfg.num_minibatches, -1)

        # Scan over minibatch indices for updates
        agent_state, losses = jax.lax.scan(
            minibatch_update,
            agent_state,
            indices
        )

        return agent_state

    def train(self) -> None:
        """Train PPO agent."""
        rng = jax.random.PRNGKey(self.cfg.seed)

        self.split = jax.jit(jax.random.split, static_argnums=1)
        self.build_batch = jax.jit(lambda transitions: jax.tree.map(lambda *x: jnp.stack(x, axis=0), *transitions))
        self.select_action = jax.jit(self.select_action)
        self.learn = jax.jit(self.learn, static_argnames="cfg")

        # Vectorised reset, get initial observations
        self.current_observations, _ = self.envs.reset(seed=self.cfg.seed)
        last_checkpoint_time = time.time()

        # Compute number of rollouts to reach total steps
        total_rollouts = self.cfg.total_steps // (self.cfg.rollout_steps * self.cfg.num_envs)
        for rollout_idx in range(total_rollouts):
            rng, key_rollout, key_learn = self.split(rng, 3)

            # Gather experience with current policy
            experience = self.rollout(key_rollout)
            batch = self.build_batch(experience)

            # Learning step
            self.agent_state = self.learn(key_learn, self.agent_state, batch, self.cfg)

        self.envs.close()
