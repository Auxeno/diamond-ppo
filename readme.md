<div align="center">

  <h1> 💎 Diamond PPO </h1>
  
  <h3>A lightweight PyTorch PPO implementation for research and experimentation</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auxeno/diamond-ppo/blob/main/notebooks/diamond-ppo-demo.ipynb)

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

Provide your own network architecture by passing an `nn.Module` network class that has the following attributes and signatures:

```python

class CustomNetwork(nn.Module):
    actor_out_layer: nn.Linear  # optional, will scale weight variance down

    def __init__(self, observation_space: Space, action_space: Space, cfg: PPOConfig) -> None: ...
    
    def get_actions(self, observations: np.ndarray, device: torch.device) -> np.ndarray: ...

    def get_values(self, observations: torch.Tensor) -> torch.Tensor: ...
    
    def get_logits_and_values(self, observations: torch.Tensor) -> torch.Tensor: ...

# Pass custom network to PPO constructor
agent = PPO(
    env_fn=lambda: gym.make("CartPole-v1"),
    network_cls=CustomNetwork
)
```

Note that PPO, ContinuousPPO and RecurrentPPO network methods have different signatures, be sure to match the network method signatures of the respective algorithm.

---

## Utilities

Optional utilities for training and debugging:

### Ticker
Live training progress display with episode statistics:
```python
from diamond.utils import Ticker
ticker = Ticker(total_steps=1_000_000, num_envs=8, rollout_steps=32)
ticker.tick(rewards, dones)

>>> Progress  |       Step  |   Episode  |  Mean Rew  |  Mean Len  |     FPS  |      Time
>>>     4.1%  |      2,040  |        84  |     23.78  |      23.8  |    2138  |  00:00:00
>>>     9.2%  |      4,600  |       140  |     36.62  |      36.6  |    1559  |  00:00:02
>>>    14.3%  |      7,160  |       180  |     51.02  |      51.0  |    1808  |  00:00:04
...

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
