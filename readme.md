<div align="center">

  <h1> 💎 Diamond PPO </h1>
  
  <h3>A lightweight PyTorch PPO implementation for research and experimentation</h3>
  
  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

---

Diamond PPO aims to be a clean and minimal PyTorch implementation of Proximal Policy Optimisation ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347)), designed for research and experimentation.  

- Built for modern Gymnasium environments (with full truncation support)
- Easy integration with your own custom neural networks
- Reproducibility via RNG seeding
- Optional lightweight utilities for logging, profiling, and checkpointing

---

## Install

> **Note:** Diamond PPO requires Python 3.10 or higher.

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

## Configuration

All training options can be changed through `PPOConfig`:

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

`PPOConfig` is documented in `ppo.py`.

---

## Custom Network

Custom networks are fully supported, just provide `actor` and `critic` methods, or modify `ppo.py` if your model structure differs.

```python
class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = NatureCNN()
        self.actor_head = nn.LazyLinear(6)
        self.critic_head = nn.LazyLinear(1)

    # Define actor, critic and forward methods
    def actor(self, x):
        x = self.base(x)
        return self.actor_head(x)

    def critic(self, x):
        x = self.base(x)
        return self.critic_head(x).squeeze(-1)

    def forward(self, x):
        x = self.base(x)
        return self.actor_head(x), self.critic_head(x).squeeze(-1)

agent = PPO(lambda: gym.make("ALE/Pong-v5"), custom_network=AtariNet())
agent.train()
```

---

## Timing

Conveniently profile sections of your code with the `Timer` utility:

```python
from utils import Timer

timer = Timer()

with timer.time("action selection"):
   action = select_action(observations)

with timer.time("env step"):
   result = env.step(action)

timer.plot_timings()
```

---

## Logging

Log data similarly to Tensorboard with the `Logger`:

```python
from utils import Logger

logger = Logger()

for step in range(num_steps):
    reward = ...  # compute or obtain reward
    logger.log("episode_reward", step, reward)

logger.plot("episode_reward")
