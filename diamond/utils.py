"""
Utility classes:
- Ticker: prints live training progress to console
- Logger: logging and plotting of training metrics
- Timer: easily profile sections of code
- Checkpointer: save and load model training state
"""
import time
from pathlib import Path
from collections import deque
from contextlib import contextmanager
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch


class Ticker:
    """
    Used for printing live training progress.
    Tracks metrics like episode length, return and FPS.

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
    >>> ticker = Ticker(total_steps=1_000_000, num_envs=16, rollout_steps=128)

    >>> for _ in range(rollout_steps):
    >>>     ...
    >>>     _, rewards, terminations, truncations, _ = envs.step(actions)
    >>>     ticker.tick(rewards, dones)

    >>> final_stats = ticker.logs
    """
    def __init__(
        self,
        total_steps: int,
        num_envs: int,
        rollout_steps: int,
        *,
        window_size: int = 100,
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

    def tick(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        **custom_logs: Any
    ) -> None:
        """Add one vector-env step and any extra scalars to the ticker."""
        self.current_step += self.num_envs
        self.current_returns += rewards
        self.current_lengths += 1

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

    def _make_checkpoints(self, num_checkpoints: int) -> np.ndarray:
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


# Custom dark template for plotly
pio.templates["umbra"] = go.layout.Template(
    layout={
        "paper_bgcolor": "#202020",
        "plot_bgcolor": "#212121",
        "font": {"color": "#f0f0f0", "size": 15},
        "title": {
            "font": {"size": 24, "color": "#f0f0f0"},
            "x": 0.04,
            "xanchor": "left",
        },
        "colorway": [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ],
        "xaxis": {
            "gridcolor": "#353535",
            "linecolor": "#909090",
            "tickcolor": "#909090",
            "zerolinecolor": "#505050",
            "title_standoff": 8,
            "tickfont": {"size": 14, "color": "#d0d0d0"},
            "ticklen": 6,
        },
        "yaxis": {
            "gridcolor": "#353535",
            "linecolor": "#909090",
            "tickcolor": "#909090",
            "zerolinecolor": "#505050",
            "title_standoff": 10,
            "tickfont": {"size": 12, "color": "#d0d0d0"},
            "ticklen": 6,
        },
        "legend": {
            "bgcolor": "#363636",
            "bordercolor": "#505050",
            "borderwidth": 0,
            "font": {"color": "#ececec", "size": 15},
        },
        "hovermode": "x unified",
    }
)  # type: ignore


class Logger:
    """
    Lightweight, torch.TensorBoard-style scalar logger and Plotly visualizer.

    Allows logging of named scalar values over steps (e.g., rewards, losses) and provides simple, interactive plots for inspection.

    Example
    -------
    >>> logger = Logger()
    >>> for step in range(num_steps):
    >>>     reward = ...  # compute or obtain reward
    >>>     logger.log("episode_reward", step, reward)
    >>> logger.plot("episode_reward")
    """
    def __init__(self):
        self.logs: dict[str, dict[str, list[Any]]] = {}
        self.theme = "umbra"

    def log(self, log_name: str, step: int, value: Any):
        """
        Log a single value.

        Parameters
        ----------
        name: str
            Metric/tag name (e.g. 'Episode Reward')
        step: int
            Global step (wall-clock, env step, episode idx, etc.)
        value: Any
            Numeric scalar
        """
        if log_name not in self.logs:
            self.logs[log_name] = {"steps": [], "values": []}
        self.logs[log_name]["steps"].append(step)
        self.logs[log_name]["values"].append(value)

    @staticmethod
    def _subsample(x: np.ndarray, y: np.ndarray, max_samples: int | None, mode: str = "uniform"):
        """Uniformly drop points so Plotly remains responsive."""
        if max_samples is None or len(x) <= max_samples:
            return x, y
        if mode == "uniform":
            idx = np.sort(np.random.choice(len(x), max_samples, replace=False))
            return x[idx], y[idx]
        raise ValueError(f"Unknown subsample_mode: {mode}")

    def plot(
        self,
        log_name: str,
        mode: str = "line",
        scale: str = "linear",
        max_samples: int | None = 10_000,
        subsample_mode: str = "uniform",
    ):
        """
        Quick visualiser (line with smoothing slider or scatter).

        Parameters
        ----------
        log_name: str
            Which metric to plot.
        mode: 'line' | 'scatter'
            Plotting mode, supported modes listed above.
        scale: 'linear' | 'log'
            y-axis scale.
        max_samples: int | None
            Cap max samples for performance (None = plot all).
        subsample_mode: str
            Currently 'uniform' only.
        """
        assert log_name in self.logs, f"No log called {log_name!r}"
        steps = np.asarray(self.logs[log_name]["steps"])
        values = np.asarray(self.logs[log_name]["values"])

        fig = go.Figure()

        if mode == "line":
            if len(values.shape) != 1:
                raise ValueError(
                    f"Log: {log_name} has data of shape: {values.shape} which "
                    "is is incompatible with mode='line'."
                )

            # Down-sample for interactivity
            x_plot, y_plot = self._subsample(steps, values, max_samples, subsample_mode)

            # Smoothing windows
            windows = [1, 5, 20, 100, 500, 2000, 10_000]

            # Convolve once on full data, then sample
            smoothed_full = [
                values if w == 1 else np.convolve(values, np.ones(w) / w, mode="same") for w in windows
            ]

            # Common subsampling indices
            idx_sub = (
                np.arange(len(steps))
                if (max_samples is None or len(steps) <= max_samples)
                else np.sort(np.random.choice(len(steps), max_samples, replace=False))
            )
            x_sub = steps[idx_sub]
            raw_sub = values[idx_sub]
            smooth_sub = [s[idx_sub] for s in smoothed_full]

            # Background raw trace (low opacity)
            fig.add_trace(
                go.Scatter(
                    x=x_sub,
                    y=raw_sub,
                    mode="lines",
                    line=dict(width=1, color="#CCCCCC"),
                    opacity=0.15,
                    showlegend=False,
                )
            )
            # Foreground smoothed variants
            for i, sm in enumerate(smooth_sub):
                fig.add_trace(
                    go.Scatter(
                        x=x_sub,
                        y=sm,
                        mode="lines",
                        line=dict(width=2),
                        opacity=1.0,
                        showlegend=False,
                        visible=(i == 0),  # default at first window
                    )
                )

            # Interactive slider
            steps_slider = []
            for i, w in enumerate(windows):
                visible = [True] + [j == i for j in range(len(windows))]
                steps_slider.append(
                    dict(method="update", args=[{"visible": visible}], label=str(w))
                )
            fig.update_layout(
                sliders=[
                    dict(
                        active=0,
                        currentvalue={"prefix": "Smoothing: "},
                        pad={"t": 0, "b": 0, "r": 20, "l": 0},
                        steps=steps_slider,
                        x=0.67,
                        y=1.27,
                        len=0.3,
                        font=dict(color="white", size=10),
                        bgcolor="#222222",
                        bordercolor="#444444",
                        borderwidth=1,
                    )
                ],
                showlegend=False,
            )

        elif mode == "scatter":
            if len(values.shape) != 1:
                raise ValueError(
                    f"Log: {log_name} has data of shape: {values.shape} which "
                    "is is incompatible with mode='scatter'."
                )

            # Down-sample for interactivity
            x_plot, y_plot = self._subsample(steps, values, max_samples, subsample_mode)

            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="markers",
                    name=log_name,
                    marker=dict(size=4, opacity=0.7),
                )
            )
        else:
            raise ValueError(f"Unknown mode {mode!r}; use 'line' or 'scatter'.")

        # Axis scaling/styling
        fig.update_layout(yaxis=dict(type=scale))
        fig.update_layout(
            template=self.theme,
            title=log_name,
            height=420,
            width=960,
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis_title="Step",
            yaxis_title=None,
        )
        fig.show()


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

        # Create Plotly bar plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=sorted_names,
                y=sorted_total_times,
                text=[f'{t:.4f}s' for t in sorted_total_times],
                textposition='outside',
                textfont=dict(size=12, color='#d0d0d0'),
                marker=dict(color='#636EFA'),
                showlegend=False,
            )
        )
        
        fig.update_layout(
            template="umbra",
            title="Code Timings",
            height=420,
            width=960,
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis_title=None,
            yaxis_title="Total Time (seconds)",
            xaxis=dict(tickangle=-45),
        )
        
        fig.show()


class Checkpointer:
    """
    Minimal utility for saving and loading model training state.

    Parameters
    ----------
    folder : str or Path, default "models"
        Directory where checkpoints are stored.
    run_name : str, default "run"
        Prefix for checkpoint filenames:  `{run_name}-stepXXXXXX.pt`.
    keep_last : int or None, default None
        Keep only the newest *k* checkpoints.
        Set to `None` (default) to keep all.

    Examples
    --------
    >>> checkpointer = Checkpointer(folder="models", run_name="test")

    Manual / fixed-interval saving
    >>> for step in range(1, total_steps + 1):
    ...     ...
    >>>     if step % 100 == 0:
    >>>         checkpointer.save(step, model, optimizer)

    Resuming later
    >>> checkpointer.load("models/test-step100.pt", model, optimizer)
    """
    def __init__(
        self,
        folder: str | Path = "models",
        run_name: str = "run",
        *,
        keep_last: int | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.run_name = run_name
        self.keep_last = keep_last

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Save a checkpoint: model (+ optimiser if given)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.run_name}-step{step:06d}.pt"
        path = self.folder / file_name
        payload = {"step": step, "model_state": model.state_dict()}

        if optimizer is not None:
            payload["opt_state"] = optimizer.state_dict()

        torch.save(payload, path)
        self._trim_old()

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Load checkpoint into model (and optimizer if given)."""
        chk = torch.load(path, map_location="cpu")
        model.load_state_dict(chk["model_state"])
        if optimizer is not None and "opt_state" in chk:
            optimizer.load_state_dict(chk["opt_state"])

    def _trim_old(self) -> None:
        if self.keep_last is None:
            return
        ckpts = sorted(self.folder.glob(f"{self.run_name}-step*.pt"))
        for old in ckpts[:-self.keep_last]:
            old.unlink(missing_ok=True)
