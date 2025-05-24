"""
Utility classes.
"""
import time
import numpy as np


class Logger:
    """Used to track episode lengths, returns, and total steps."""
    def __init__(
        self, 
        total_steps: int, 
        num_envs: int,
        rollout_steps: int,
        *,
        window_size: int = 20,  # Number of recent episodes to consider for averages
        print_every: int = 5,  # Print logs after this many vectorised steps
        num_checkpoints: int = 20,
    ):

        # Calculate when to make new lines
        steps_per_rollout = rollout_steps * num_envs
        total_iterations: int   = total_steps // steps_per_rollout
        self.checkpoints = (np.arange(1, num_checkpoints + 1) * 
            total_iterations // num_checkpoints) * steps_per_rollout

        # Step and episode counters
        self.current_step = 0
        self.current_episode = 1

        # Arrays to track returns and lengths per environment
        self.current_returns = np.zeros(num_envs, dtype=np.float32)
        self.current_lengths = np.zeros(num_envs, dtype=np.int64)

        # Lists to store completed episode statistics
        self.episode_returns = []
        self.episode_lengths = []

        # Custom logging structures
        self.custom_logs = {}
        self.custom_log_keys = []

        # Initialise timing and checkpointing
        self.start_time = time.time()
        self.total_steps = total_steps
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.num_checkpoints = num_checkpoints
        self.print_every = print_every
        self.window_size = window_size
        self.last_checkpoint_time = self.start_time
        self.last_checkpoint_step = 0
        self.header_printed = False        
        
    def reset(self):
        self.__init__(self.total_steps, self.num_envs, 
                      self.rollout_steps, self.num_checkpoints)
    
    def log(
        self, 
        rewards: float,
        terminations: bool, 
        truncations: bool, 
        **kwargs
    ):
        """Updates logger with latest rewards, done flags and any custom logs."""
        # Update steps and returns for vectorised environments
        self.current_step += self.num_envs
        self.current_returns += rewards
        self.current_lengths += 1
        
        # Handle environments being done
        dones = np.logical_or(terminations, truncations)
        done_returns = self.current_returns[dones]
        done_lengths = self.current_lengths[dones]
        for ep_return, ep_length in zip(done_returns, done_lengths):
            self.episode_returns.append(ep_return)
            self.episode_lengths.append(ep_length)
            self.current_episode += 1
        self.current_returns = np.where(dones, 0.0, self.current_returns)
        self.current_lengths = np.where(dones, 0, self.current_lengths)

        # Update custom_logs with any additional keyword arguments
        for key, value in kwargs.items():
            if key not in self.custom_log_keys:
                self.custom_log_keys.append(key)
            self.custom_logs[key] = value

    def print_logs(self):
        """Prints training progress with headers and updates."""
        if (self.current_step % self.num_envs * self.print_every == 0
            and len(self.episode_returns) > 0):
            elapsed_time = time.time() - self.start_time
            
            # FPS based on last checkpoint
            steps_since_checkpoint = self.current_step - self.last_checkpoint_step
            time_since_checkpoint = time.time() - self.last_checkpoint_time
            fps = steps_since_checkpoint / time_since_checkpoint if time_since_checkpoint > 0 else 0

            # Calculate other metrics
            progress = 100 * self.current_step / self.total_steps
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
                # Append custom log headers
                for key in self.custom_log_keys:
                    log_header += f"  |  {key:>{len(key)}}"
                print(log_header)
                self.header_printed = True

            log_string = (
                f"{progress:>7.1f}%  |  "
                f"{self.current_step:>9,}  |  "
                f"{self.current_episode:>8,}  |  "
                f"{mean_reward:>8.2f}  |  "
                f"{mean_ep_length:>8.1f}  |  "
                f"{fps:>6,.0f}  |  "
                f"{formatted_time:>8}"
            )
            # Append custom log values
            for key in self.custom_log_keys:
                value = self.custom_logs.get(key, 0)
                # Format based on the type of value
                if isinstance(value, float):
                    log_string += f"  |  {value:>{len(key)}.2f}"
                elif isinstance(value, int):
                    log_string += f"  |  {value:>{len(key)}d}"
                else:
                    log_string += f"  |  {str(value):>{len(key)}}"
            print(f"\r{log_string}", end='')
        
        # Check if a checkpoint is reached
        if self.current_step in self.checkpoints:
            print()
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = self.current_step

    @property
    def logs(self):
        return  {
            'total_steps': self.current_step,
            'total_episodes': self.current_episode - 1,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'best_reward': np.max(self.episode_returns) if len(self.episode_returns) > 0 else None,
            'total_duration': time.time() - self.start_time,
            'mean_fps': self.current_step / (time.time() - self.start_time + 1e-6),
            'custom_logs': self.custom_logs
        }
    

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
