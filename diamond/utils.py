"""
Utility classes.
"""
class Logger:
    """Helper class used to save and print metrics during training."""
    def log(self, actions, rewards, terminations, truncations):
        """Logs results of a transition."""
        pass

class Timer:
    """Convenient way to profile code execution."""
    pass

class Checkpointer:
    """Convenience class used to save agents to disk."""
    def __init__(self, folder="models", run_name="Test"):
        pass

    def save(self):
        """Saves network weights to disk."""
        pass
