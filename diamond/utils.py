"""
Utility classes.
"""
import time
import numpy as np


class Logger:
    """Helper class used to save and print metrics during training."""
    def __init__(
        self, 
        total_steps: int, 
        num_envs: int, 
        log_interval: int = 10,
        log_lines: int = 20,
        window_size: int = 20,
    ):
        # Step and episode counters
        self.current_step = 0
        self.done_episodes = 0

        # Arrays and lists to track returns and lengths per environment
        self.current_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_returns = []
        self.current_lengths = np.zeros(num_envs, dtype=np.int64)
        self.episode_lengths = []

        self.start_time = time.time()
        self.header_printed = False
    
        self.total_steps = total_steps
        self.num_envs = num_envs
        self.log_interval = log_interval
        self.log_lines = log_lines
        self.window_size = window_size

    def log(
        self, 
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray
    ) -> None:
        """Logs results of a transition."""
        self.current_step += self.num_envs
        self.current_returns += rewards
        self.current_lengths += 1

        # Done environments
        dones = np.logical_or(terminations, truncations)
        done_returns = self.current_returns[dones]
        done_lengths = self.current_lengths[dones]
        for ep_return, ep_length in zip(done_returns, done_lengths):
            self.episode_returns.append(ep_return)
            self.episode_lengths.append(ep_length)
            self.done_episodes += 1
        self.current_returns = np.where(dones, 0.0, self.current_returns)
        self.current_lengths = np.where(dones, 0, self.current_lengths)

    def reset(self):
        self.__init__(self.total_steps, self.num_envs)

    def print_logs(self):
        if (self.current_step % self.log_interval * self.num_envs == 0 
            and len(self.episode_returns) > 0):
            elapsed_time = time.time() - self.start_time

            progress = 100.0 * self.current_step / self.total_steps
            if len(self.episode_returns) >= self.window_size:
                mean_reward = np.mean(self.episode_returns[-self.window_size:])
                mean_ep_length = np.mean(self.episode_lengths[-self.window_size:])
            else:
                mean_reward = np.mean(self.episode_returns)
                mean_ep_length = np.mean(self.episode_lengths)
            
            # Format elapsed time into hh:mm:ss
            hours, remainder = divmod(int(elapsed_time), 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"

            if not self.header_printed:
                log_header = (
                    f"{'Progress':>8}  |  "
                    f"{'Step':>9}  |  "
                    f"{'Episode':>8}  |  "
                    f"{'Mean Rew':>8}  |  "
                    f"{'Mean Len':<7}  |  "
                    f"{'FPS':>6}  |  "
                    f"{'Time':>8}"
                )
                print(log_header)
                self.header_printed = True
            
            log_string = (
                f"{progress:>7.1f}%  |  "
                f"{self.current_step:>9,}  |  "
                f"{self.done_episodes:>8,}  |  "
                f"{mean_reward:>8.2f}  |  "
                f"{mean_ep_length:>8.1f}  |  "
                f"{100.3:>6,.0f}  |  "
                f"{formatted_time:>8}"
            )

            print(f"\r{log_string}", end='')


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
