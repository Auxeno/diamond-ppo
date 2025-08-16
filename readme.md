<div align="center">

  <h1> 💎 Diamond PPO </h1>
  
  <h3>A lightweight PyTorch PPO implementation for research and experimentation</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Diamond PPO is a clean, minimal PyTorch implementation of Proximal Policy Optimisation ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)) designed for research and experimentation.  

- **Multiple PPO variants:** Standard (discrete), Continuous, and Recurrent
- **Modern Gymnasium environments** with full truncation support
- **Custom neural networks** via simple interface
- **Lightweight utilities** for logging, profiling, and checkpointing

---

## Install

> **Note:** Diamond PPO requires Python 3.10 or higher.

Install with `pip`:
```bash
pip install git+https://github.com/auxeno/diamond-ppo
```

Or clone for development:
```bash
git clone https://github.com/auxeno/diamond-ppo
cd diamond-ppo
pip install -r requirements.txt
```

---

## Quick Start

```python
from diamond import PPO

agent = PPO(env_fn=lambda: gym.make("CartPole-v1"))
agent.train()
```

### PPO Variants

```python
# Standard PPO for discrete action spaces
from diamond import PPO, PPOConfig

# Continuous PPO for continuous control
from diamond import ContinuousPPO, ContinuousPPOConfig

# Recurrent PPO (discrete)
from diamond import RecurrentPPO, RecurrentPPOConfig
```

---

## Configuration

Each PPO variant has its own config class with all hyperparameters documented:

```python
from diamond import PPO, PPOConfig

config = PPOConfig(
    total_steps=1_000_000,
    rollout_steps=128,
    num_envs=8,
    checkpoint=True,
    save_interval=300
)

agent = PPO(lambda: gym.make("LunarLander-v3"), cfg=config)
agent.train()
```

---

## Custom Networks

Provide your own network architecture by implementing the required interface:

```python
class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = NatureCNN()
        self.actor_head = nn.LazyLinear(6)
        self.critic_head = nn.LazyLinear(1)

    def actor(self, x):
        """Returns action logits/parameters"""
        x = self.base(x)
        return self.actor_head(x)

    def critic(self, x):
        """Returns value estimates"""
        x = self.base(x)
        return self.critic_head(x).squeeze(-1)

    def forward(self, x):
        """Returns both actor and critic outputs"""
        x = self.base(x)
        return self.actor_head(x), self.critic_head(x).squeeze(-1)

agent = PPO(lambda: gym.make("ALE/Pong-v5"), custom_network=AtariNet())
agent.train()
```

Custom networks must implement all three methods (`actor`, `critic`, `forward`) matching the interface used by the default network.

---

## Utilities

Optional utilities for training and debugging:

### Ticker
Live training progress display with episode statistics:
```python
from diamond.utils import Ticker
ticker = Ticker(total_steps=1_000_000)
```

### Logger
Metrics logging and plotting:
```python
from diamond.utils import Logger
logger = Logger()
logger.log("episode_reward", step, reward)
logger.plot("episode_reward")
```

### Timer
Code profiling with context managers:
```python
from diamond.utils import Timer
timer = Timer()
with timer.time("env step"):
    result = env.step(action)
timer.plot_timings()
```

### Checkpointer
Automatic model saving and loading:
```python
from diamond.utils import Checkpointer
checkpointer = Checkpointer(model, optimizer, save_dir="models")
