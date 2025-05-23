"""
Configuration for diamond PPO.
"""

config = {
    "network": {
        "hidden_sizes": [64, 64],
        "hidden_activation": "relu"
    },
    "ppo": {
        "ppo_clip": 0.2,
        "gae_lambda": 0.95
    }
}