<div align="center">

  <h1> 💎 Diamond PPO </h1>
  
  <h3>A lightweight PyTorch PPO implementation for research and experimentation</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Diamond PPO aims to be a clean and minimal implementation of Proximal Policy Optimisation written in PyTorch.  

It is designed for Gymnasium environments and is compatible with custom user-defined neural networks.

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

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor  = nn.Linear(obs_dim, act_dim)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.actor(x), self.critic(x)

# Example usage (CartPole)
model = MyNet(obs_dim=4, act_dim=2)

agent = PPO(env_fn, custom_network=model)
agent.train()
```

---

## Timing

Conveniently profile sections of your code with the `Timer` utility:

```python
>>> with timer.time("action selection"):
>>>    action = select_action(observations)

>>> with timer.time("env step"):
>>>    result = env.step(action)

>>> timer.plot_timings()
```

---

## Licence

This project is licensed under the Apache 2.0 License.

> Copyright © 2025
> Alex – https://github.com/auxeno
