"""
Utility classes.
"""
from typing import Any
from contextlib import contextmanager
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


class Logger:
    """
    Track episode lengths, returns, steps and print training progress.

    Parameters
    ----------
    total_steps : int
        Total number of steps the training process will run.
    num_envs : int
        Number of vectorised environments.
    rollout_steps : int
        Number of steps per rollout.
    window_size : int, optional
        Number of recent episodes to average when computing stats, by default 20.
    print_every : int, optional
        Number of environment steps between prints (scaled by num_envs), by default 5.
    num_checkpoints : int, optional
        Number of line breaks/checkpoints in the printed output, by default 20.
    verbose : bool, optional
        Whether to automatically print logs when logging steps, by default True.

    Example
    -------
    >>> logger = Logger(total_steps=1_000_000, num_envs=16, rollout_steps=128)

    >>> for _ in range(rollout_steps):
    >>>     ...
    >>>     _, rewards, terminations, truncations, _ = envs.step(actions)
    >>>     logger.log(rewards, terminations, truncations)

    >>> final_stats = logger.logs
    """
    def __init__(
        self,
        total_steps: int,
        num_envs: int,
        rollout_steps: int,
        *,
        window_size: int = 20,
        print_every: int = 5,
        num_checkpoints: int = 20,
        verbose: bool = True
    ):

        # Fixed state
        self.total_steps = total_steps
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.window_size = window_size
        self.print_every = print_every
        self.verbose = verbose
        self.checkpoints = self._make_checkpoints(num_checkpoints)

        # Running state
        self.current_step = 0
        self.current_episode = 1
        self.current_returns = np.zeros(num_envs, np.float32)
        self.current_lengths = np.zeros(num_envs, np.int64)
        self.recent_returns: deque[float] = deque(maxlen=window_size)
        self.recent_lengths: deque[int] = deque(maxlen=window_size)
        self.custom_logs: dict[str, Any] = {}
        self._header_printed = False
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_step = 0

    def reset(self) -> None:
        """Clear counters and timing but keep the same configuration."""
        self.__init__(
            self.total_steps,
            self.num_envs,
            self.rollout_steps,
            window_size=self.window_size,
            print_every=self.print_every,
            num_checkpoints=len(self.checkpoints),
            verbose=self.verbose
        )

    def log(
        self,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        **custom_logs: Any,
    ) -> None:
        """Add one vector-env step and any extra scalars to the logger."""
        self.current_step += self.num_envs
        self.current_returns += rewards
        self.current_lengths += 1

        dones = np.logical_or(terminations, truncations)
        for ep_return, ep_length in zip(
            self.current_returns[dones], self.current_lengths[dones]
        ):
            self.recent_returns.append(float(ep_return))
            self.recent_lengths.append(int(ep_length))
            self.current_episode += 1

        self.current_returns[dones] = 0.0
        self.current_lengths[dones] = 0

        self.custom_logs.update(custom_logs)

        if self.verbose:
            self.print_logs()

    def print_logs(self) -> None:
        """Print a progress row when the interval or checkpoint is hit."""
        if self.current_step in self.checkpoints:
            if self._header_printed:
                print()
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = self.current_step

        if not self._should_print():
            return

        elapsed = time.time() - self.start_time
        progress = 100 * self.current_step / self.total_steps

        mean_reward = np.mean(self.recent_returns)
        mean_length = np.mean(self.recent_lengths)

        steps_since_checkpoint = self.current_step - self.last_checkpoint_step
        time_since_checkpoint = time.time() - self.last_checkpoint_time
        fps = steps_since_checkpoint / (time_since_checkpoint + 1e-6)

        time_str = self._format_hms(elapsed)

        if not self._header_printed:
            self._print_header()
            self._header_printed = True

        row = (
            f"{progress:>7.1f}%  |  "
            f"{self.current_step:>9,}  |  "
            f"{self.current_episode:>8,}  |  "
            f"{mean_reward:>8.2f}  |  "
            f"{mean_length:>8.1f}  |  "
            f"{fps:>6.0f}  |  "
            f"{time_str:>8}"
        )
        for key in self.custom_logs:
            val = self.custom_logs[key]
            if isinstance(val, float):
                row += f"  |  {val:.2f}"
            else:
                row += f"  |  {val}"

        print("\r" + row, end="")

    @property
    def logs(self) -> dict[str, Any]:
        """Return a summary dictionary of accumulated statistics."""
        elapsed = time.time() - self.start_time
        return dict(
            total_steps=self.current_step,
            total_episodes=self.current_episode - 1,
            episode_returns=list(self.recent_returns),
            episode_lengths=list(self.recent_lengths),
            best_reward=max(self.recent_returns, default=None),
            total_duration=elapsed,
            mean_fps=self.current_step / (elapsed + 1e-6),
            custom_logs=self.custom_logs.copy(),
        )

    def _make_checkpoints(self, num_checkpoints: int) -> list[int]:
        steps_per_rollout = self.rollout_steps * self.num_envs
        total_iterations = self.total_steps // steps_per_rollout
        return (np.arange(1, num_checkpoints + 1) * 
            total_iterations // num_checkpoints) * steps_per_rollout

    def _should_print(self) -> bool:
        step_interval = self.num_envs * self.print_every
        return (
            self.current_step % step_interval == 0
            and len(self.recent_returns) > 0
        )

    def _format_hms(self, seconds_total: float) -> str:
        h, rem = divmod(int(seconds_total), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02}:{m:02}:{s:02}"

    def _print_header(self) -> None:
        header = (
            f"{'Progress':>8}  |  "
            f"{'Step':>9}  |  "
            f"{'Episode':>8}  |  "
            f"{'Mean Rew':>8}  |  "
            f"{'Mean Len':>7}  |  "
            f"{'FPS':>6}  |  "
            f"{'Time':>8}"
        )
        for key in self.custom_logs:
            header += f"  |  {key}"
        print(header)

class Timer:
    """
    Profile sections of code using context manager blocks.

    Example
    -------
    >>> timer = Timer()

    >>> with timer.time("action selection"):
    >>>    action = agent.select_action(observations)

    >>> with timer.time("env step"):
    >>>    result = env.step(action)

    >>> timer.plot_timings()
    """
    def __init__(self):
        self.timings = {}
    
    def reset(self) -> None:
        """Reset all recorded timings."""
        self.__init__()
    
    @contextmanager
    def time(self, name: str):
        """Time a code block and record its average over repeated calls."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed_time = time.time() - start_time
            
            if name not in self.timings:
                self.timings[name] = {'avg_time': elapsed_time, 'count': 1}
            else:
                timing = self.timings[name]
                timing["count"] += 1
                timing["avg_time"] += (elapsed_time - timing["avg_time"]) / timing["count"]
    
    def plot_timings(self) -> None:
        """Plot total time spent in each labelled block."""
        if not self.timings:
            print("No timings to plot.")
            return

        # Extract names and compute total times
        names = list(self.timings.keys())
        total_times = [data["avg_time"] * data["count"] for data in self.timings.values()]

        # Sort the timings by total time in descending order for better visualisation
        sorted_indices = sorted(range(len(total_times)), key=lambda i: total_times[i], reverse=True)
        sorted_names = [names[i] for i in sorted_indices]
        sorted_total_times = [total_times[i] for i in sorted_indices]

        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_names, sorted_total_times, color='#636EFA')

        # Add text labels above the bars
        for bar, total_time in zip(bars, sorted_total_times):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                     f'{total_time:.4f}s', ha='center', va='bottom', fontsize=8)

        plt.ylabel('Total Time (seconds)')
        plt.title('Code Timings')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

class Checkpointer:
    """Convenience class used to save agents to disk."""
    def __init__(self, folder="models", run_name="Test"):
        pass

    def save(self):
        """Saves network weights to disk."""
        pass
