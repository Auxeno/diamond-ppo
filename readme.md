<div align="center">

  <h1> 💎 Diamond PPO </h1>
  
  <h3>A lightweight PyTorch PPO implementation for research and experimentation</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Aims to be a clean and minimal PyTorch implementation of Proximal Policy Optimisation ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)).  

Designed for Gymnasium environments and compatible with custom user-defined neural networks.

Includes optional utilities for logging, timing, and checkpointing.

---

## Features

- Minimal, readable implementation (~300 lines)
- Supports custom networks
- Vectorised environments
- Full-support for truncation

---

## Install

> **Note:** Diamond PPO requires Python 3.10 or higher (tested on 3.12).

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

---

## Custom Network

Custom networks are fully supported, just provide `actor` and `critic` attributes, or modify `ppo.py` if your model structure differs.

```python
class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = NatureCNN()
        self.actor_head = nn.LazyLinear(6)
        self.critic_head = nn.LazyLinear(1)

        # Define actor and critic attributes
        self.actor = nn.Sequential(self.base, self.actor_head)
        self.critic = nn.Sequential(self.base, self.critic_head)

    def forward(self, x):
        x = self.base(x)
        return self.actor_head(x), self.critic_head(x)

# Example usage (Pong)
agent = PPO(lambda: gym.make("ALE/Pong-v5"), custom_network=AtariNet())
agent.train()
```

---

## Timing

Conveniently profile sections of your code with the `Timer` utility:

```python
with timer.time("action selection"):
   action = select_action(observations)

with timer.time("env step"):
   result = env.step(action)

timer.plot_timings()
```

