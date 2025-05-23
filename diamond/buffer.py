from dataclasses import dataclass, asdict
import numpy as np
import torch


@dataclass
class Transition:
    """
    Holds results of vectorised environment step.
    """
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray

    def __iter__(self):
        """Allows for iteration over Dataclass objects"""
        return iter(asdict(self).items())

class Buffer:
    """
    Fixed-size storage for a full rollout.

    Pre-allocates NumPy arrays with shape (T, *sample.shape)
    Attribute-style access: `buffer.rewards`  or  `buffer.actions`
    Explicit write index: `buffer.store(t, transition)`

    Parameters
    ----------
    sample: Transition
        Result from a vectorised environment step.
    num_steps: int
        How many timesteps (T) the rollout will contain.
    """
    def __init__(
        self,
        sample: dict,
        num_steps: int
    ):
        if not isinstance(sample, Transition):
            raise TypeError("Buffer expects a Transition instance as template")

        self._data: dict[str, np.ndarray] = {}
        for name, sample_array in sample:
            shape, dtype = sample_array.shape, sample_array.dtype
            self._data[name] = np.zeros((num_steps, *shape), dtype=dtype)
            setattr(self, name, self._data[name])

    def store(self, idx: int, transition: Transition) -> None:
        """
        Write a vectorised transition for all envs at the given time index.

        Parameters
        ----------
        idx : int
            0-based timestep into the rollout.
        transition : Transition
            Data for *all* parallel envs at that timestep.
        """
        for name, value in transition:
            self._data[name][idx] = value

    def as_torch(self, device=None):
        """Returns each buffer item as a PyTorch tensor."""
        pass

    def __repr__(self) -> str:
        """
        Pretty printing for buffer objects.
        """
        def _bytes2str(b: int) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if b < 1024:
                    return f"{b:.0f} {unit}"
                b /= 1024
            return f"{b:.1f} TB"

        if not self._data:
            return "<Buffer: empty>"

        # Get sample item from buffer
        array_0 = next(iter(self._data.values()))

        # Infer number of timesteps and environments
        num_timesteps, num_environments = array_0.shape[:2]

        # Calculate memory usage
        total_bytes = sum(arr.nbytes for arr in self._data.values())

        # Build __repr__ string
        lines = [f"<Buffer>",
                f"  Timesteps : {num_timesteps}",
                f"  Environments : {num_environments}",
                f"  Memory usage : {_bytes2str(total_bytes)}",
                f"  Fields:"]
        for name, array in self._data.items():
            lines.append(f"    <{name}> shape={array.shape}, dtype={array.dtype}")
        
        return "\n".join(lines)
    