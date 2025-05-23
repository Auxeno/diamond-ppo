import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv

from .buffer import Transition, Buffer


class PPO:
    def __init__(
        self,
        env_fn: callable,
        config: dict
    ):
        # Create vectorised environments
        self.envs = SyncVectorEnv(
            [env_fn for _ in range(config["num_envs"])], 
            copy=False,
            autoreset_mode="SameStep"
        )
        self.buffer = None

        self.network = None
        self.optimizer = torch.optim.Adam(
            self.network.params, lr=config["learning_rate"]
        )

        self.logger = None
        self.checkpointer = None
        self.device = config["device"]

    def select_action(self, observations: np.ndarray) -> np.ndarray:
        observations_tensor = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )
        # Forward pass with policy network
        with torch.no_grad():
            logits = self.network.actor(observations_tensor)

        # Boltzmann action selection
        actions = torch.distributions.Categorical(logits=logits).sample()
        
        # Return as NumPy array
        return actions.cpu().numpy()

    def rollout(self) -> tuple:
        pass

    def learn(self):
        pass

    def train(self):
        pass
    