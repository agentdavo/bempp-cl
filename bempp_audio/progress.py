"""
Logging and progress tracking for bempp_audio.

Provides configurable logging and progress reporting for long-running
simulations like frequency sweeps.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Iterator, Any, Dict
from dataclasses import dataclass


# Global configuration
_config = {
    "log_mode": False,  # If True, output is log-file friendly (no \r updates)
    "timestamps": True,  # Include timestamps in log messages
    "progress_interval": 1.0,  # Minimum seconds between progress updates
    "style": "pretty",  # "pretty" or "plain"
    "show_name": False,  # Include logger name in prefix
    "stream": sys.stdout,  # Where to write logs/progress.
}


def configure(
    log_mode: Optional[bool] = None,
    timestamps: Optional[bool] = None,
    progress_interval: Optional[float] = None,
    style: Optional[str] = None,
    show_name: Optional[bool] = None,
    stream=None,
):
    """
    Configure global logging and progress settings.

    Parameters
    ----------
    log_mode : bool, optional
        If True, output is log-file friendly (summary only, no carriage returns).
    timestamps : bool, optional
        If True, include timestamps in log messages.
    progress_interval : float, optional
        Minimum seconds between progress display updates.
    """
    if log_mode is not None:
        _config["log_mode"] = log_mode
    if timestamps is not None:
        _config["timestamps"] = timestamps
    if progress_interval is not None:
        _config["progress_interval"] = progress_interval
    if style is not None:
        if style not in ("pretty", "plain"):
            raise ValueError("style must be 'pretty' or 'plain'")
        _config["style"] = style
    if show_name is not None:
        _config["show_name"] = bool(show_name)
    if stream is not None:
        _config["stream"] = stream


def _stream():
    return _config.get("stream", sys.stdout)


# Auto-detect if output is being piped/redirected (not a TTY)
if not sys.stdout.isatty():
    _config["log_mode"] = True


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    current: int
    total: int
    elapsed: float
    rate: float
    eta: float
    current_item: Any = None


class ProgressTracker:
    """
    Track progress of iterative operations with time estimates.

    Parameters
    ----------
    total : int
        Total number of items to process.
    desc : str
        Description of the operation.
    unit : str
        Unit name for items (e.g., 'freq', 'element').
    disable : bool
        If True, suppress all output.
    log_mode : bool, optional
        Override global log_mode setting.

    Examples
    --------
    >>> with ProgressTracker(100, desc="Solving") as pbar:
    ...     for i in range(100):
    ...         # do work
    ...         pbar.update(item=f"{i} Hz")

    >>> for freq in ProgressTracker.iterate(frequencies, desc="Frequency sweep"):
    ...     # process freq
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        unit: str = "it",
        disable: bool = False,
        log_mode: Optional[bool] = None,
    ):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.log_mode = log_mode if log_mode is not None else _config["log_mode"]
        self.current = 0
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.last_display_time: float = 0
        self._current_item = None
        self._last_logged_pct = -1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.close()

    def start(self):
        """Start the progress tracker."""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_display_time = self.start_time
        self.current = 0
        if not self.disable:
            if self.log_mode:
                self._log_start()
            else:
                self._display()

    def update(self, n: int = 1, item: Any = None):
        """
        Update progress by n items.

        Parameters
        ----------
        n : int
            Number of items completed.
        item : Any
            Current item being processed (for display).
        """
        self.current += n
        self._current_item = item
        self.last_update_time = time.time()

        if self.disable:
            return

        if self.log_mode:
            # In log mode, only log at certain percentages to avoid spam
            pct = int(100 * self.current / self.total) if self.total > 0 else 0
            # Log at 25%, 50%, 75%, and 100%
            if pct >= self._last_logged_pct + 25:
                self._log_progress()
                self._last_logged_pct = (pct // 25) * 25
        else:
            # Rate-limit display updates
            now = time.time()
            if now - self.last_display_time >= _config["progress_interval"]:
                self._display()
                self.last_display_time = now

    def close(self):
        """Close the progress tracker."""
        if self.disable:
            return

        if self.log_mode:
            self._log_complete()
        else:
            self._display(final=True)
            _stream().write("\n")
            _stream().flush()

    def stats(self) -> ProgressStats:
        """Get current progress statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = self.total - self.current
        eta = remaining / rate if rate > 0 else 0

        return ProgressStats(
            current=self.current,
            total=self.total,
            elapsed=elapsed,
            rate=rate,
            eta=eta,
            current_item=self._current_item,
        )

    def _log_start(self):
        """Log start message (log mode)."""
        logger = get_logger()
        logger.info(f"{self.desc}: starting ({self.total} {self.unit}s)")

    def _log_progress(self):
        """Log progress update (log mode)."""
        stats = self.stats()
        pct = 100 * stats.current / stats.total if stats.total > 0 else 0
        elapsed_str = self._format_time(stats.elapsed)
        eta_str = self._format_time(stats.eta)
        logger = get_logger()
        logger.info(
            f"{self.desc}: {stats.current}/{stats.total} ({pct:.0f}%) "
            f"[{elapsed_str} elapsed, ~{eta_str} remaining]"
        )

    def _log_complete(self):
        """Log completion message (log mode)."""
        stats = self.stats()
        elapsed_str = self._format_time(stats.elapsed)
        logger = get_logger()
        logger.info(
            f"{self.desc}: completed {stats.total} {self.unit}s in {elapsed_str} "
            f"({stats.rate:.1f} {self.unit}/s)"
        )

    def _display(self, final: bool = False):
        """Display progress bar (interactive mode)."""
        stats = self.stats()

        # Build progress bar
        pct = stats.current / stats.total if stats.total > 0 else 0
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "=" * filled + ">" + " " * (bar_width - filled - 1)
        if pct >= 1.0:
            bar = "=" * bar_width

        # Format times
        elapsed_str = self._format_time(stats.elapsed)
        eta_str = self._format_time(stats.eta) if not final else "0:00"

        # Build status line
        item_str = f" [{stats.current_item}]" if stats.current_item else ""
        status = (
            f"\r{self.desc}: |{bar}| "
            f"{stats.current}/{stats.total} "
            f"[{elapsed_str}<{eta_str}, {stats.rate:.1f} {self.unit}/s]"
            f"{item_str}"
        )

        # Pad to clear previous line
        status = status.ljust(120)

        out = _stream()
        out.write(status)
        out.flush()

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as M:SS or H:MM:SS."""
        if seconds < 0:
            return "?"
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    @classmethod
    def iterate(
        cls,
        iterable,
        desc: str = "Processing",
        unit: str = "it",
        disable: bool = False,
        total: Optional[int] = None,
        log_mode: Optional[bool] = None,
    ) -> Iterator:
        """
        Iterate with progress tracking.

        Parameters
        ----------
        iterable : iterable
            Items to iterate over.
        desc : str
            Description of the operation.
        unit : str
            Unit name for items.
        disable : bool
            If True, suppress output.
        total : int, optional
            Total count if iterable doesn't support len().
        log_mode : bool, optional
            Override global log_mode setting.

        Yields
        ------
        item
            Items from the iterable.
        """
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = 0

        tracker = cls(total, desc, unit, disable, log_mode)
        tracker.start()

        try:
            for item in iterable:
                yield item
                tracker.update(item=item)
        finally:
            tracker.close()


class Logger:
    """
    Simple logger for bempp_audio operations.

    Parameters
    ----------
    name : str
        Logger name.
    level : str
        Logging level: 'debug', 'info', 'warning', 'error', 'silent'.
    """

    LEVELS = {'debug': 0, 'info': 1, 'warning': 2, 'error': 3, 'silent': 4}
    LEVEL_NAMES = {0: 'DEBUG', 1: 'INFO', 2: 'WARN', 3: 'ERROR'}

    def __init__(self, name: str = "bempp_audio", level: str = "info"):
        self.name = name
        self.level = level
        self._indent = 0
        self._step_stack: list[int] = []
        self._step_counter = 0

    def _prefix(self, level_num: int) -> str:
        lvl = self.LEVEL_NAMES.get(level_num, "INFO")
        ts = datetime.now().strftime("%H:%M:%S") if _config["timestamps"] else ""
        ts_part = f"[{ts}] " if ts else ""
        name_part = f"{self.name}: " if _config.get("show_name", False) else ""
        indent_part = "  " * self._indent

        if _config["style"] == "plain":
            return f"{ts_part}{lvl}: {name_part}{indent_part}"

        # Pretty fixed-width prefix (log-friendly, no repeated module tag by default).
        return f"{ts_part}{lvl:<5} {name_part}{indent_part}"

    def _log(self, level: str, msg: str):
        """Internal log method."""
        level_num = self.LEVELS.get(level, 1)
        if level_num >= self.LEVELS.get(self.level, 1):
            print(f"{self._prefix(level_num)}{msg}", file=_stream())

    def debug(self, msg: str):
        """Log debug message."""
        self._log('debug', msg)

    def info(self, msg: str):
        """Log info message."""
        self._log('info', msg)

    def warning(self, msg: str):
        """Log warning message."""
        self._log('warning', msg)

    def warn(self, msg: str):
        """Log warning message (alias)."""
        self._log('warning', msg)

    def error(self, msg: str):
        """Log error message."""
        self._log('error', msg)

    def section(self, title: str, char: str = "=", width: int = 70):
        """
        Log a section header.
        
        Parameters
        ----------
        title : str
            Section title.
        char : str
            Character to use for border (default: "=").
        width : int
            Total width of header (default: 70).
        """
        border = char * width
        self.info(border)
        self.info(title)
        self.info(border)

    def subsection(self, title: str, char: str = "-", width: int = 70):
        """
        Log a subsection header.
        
        Parameters
        ----------
        title : str
            Subsection title.
        char : str
            Character to use for border (default: "-").
        width : int
            Total width of header (default: 70).
        """
        border = char * width
        self.info(border)
        self.info(title)
        self.info(border)

    def blank(self):
        """Print a blank line (for spacing)."""
        if _config.get("log_mode", False):
            return
        print("", file=_stream())

    def success(self, msg: str):
        """Log success message."""
        self.info(f"OK: {msg}")

    def failure(self, msg: str):
        """Log failure message."""
        self.error(f"FAILED: {msg}")

    def config(self, label: str, value: Any):
        """
        Log a configuration key-value pair.
        
        Parameters
        ----------
        label : str
            Configuration parameter name.
        value : Any
            Configuration value.
        """
        self.info(f"  {label}: {value}")

    def validation(self, passed: bool, msg: str):
        """
        Log a validation result.
        
        Parameters
        ----------
        passed : bool
            Whether validation passed.
        msg : str
            Validation message.
        """
        if passed:
            self.info(f"OK: {msg}")
        else:
            self.error(f"FAILED: {msg}")

    def rule(self, char: str = "─", width: int = 72):
        """Print a horizontal rule."""
        if _config["style"] == "plain":
            self.info(char * width)
        else:
            self.info(char * width)

    @contextmanager
    def group(self, title: str, *, level: str = "info") -> Iterator[None]:
        """
        Group log lines with automatic duration reporting.

        Example:
            with logger.group("Mesh generation"):
                ...
        """
        log_fn = getattr(self, level, self.info)
        start = time.time()
        log_fn(f"{title}")
        self._indent += 1
        try:
            yield
        finally:
            self._indent = max(0, self._indent - 1)
            elapsed = time.time() - start
            self.info(f"Done ({ProgressTracker._format_time(elapsed)})")

    @contextmanager
    def step(self, title: str) -> Iterator[int]:
        """
        A numbered step context for long workflows.

        Example:
            with logger.step("Assemble operators") as n:
                ...
        """
        self._step_counter += 1
        n = self._step_counter
        start = time.time()
        self.info(f"Step {n}: {title}")
        self._indent += 1
        try:
            yield n
        finally:
            self._indent = max(0, self._indent - 1)
            elapsed = time.time() - start
            self.info(f"Step {n} done ({ProgressTracker._format_time(elapsed)})")

    def config_section(self, title: str, config_dict: dict, format_fn=None):
        """
        Log a configuration section with multiple key-value pairs.
        
        Parameters
        ----------
        title : str
            Section title (e.g., "Simulation Parameters").
        config_dict : dict
            Dictionary of configuration parameters.
        format_fn : callable, optional
            Optional function to format values (e.g., lambda x: f"{x*1000:.1f} mm").
            If not provided, uses str().
        
        Examples
        --------
        >>> logger.config_section("Geometry", {
        ...     "Throat diameter": 0.025,
        ...     "Mouth diameter": 0.15,
        ...     "Length": 0.10
        ... }, lambda x: f"{x*1000:.1f} mm" if isinstance(x, (int, float)) else x)
        """
        self.info(f"{title}:")
        for key, value in config_dict.items():
            if format_fn is not None:
                try:
                    formatted_value = format_fn(value)
                except Exception:
                    formatted_value = str(value)
            else:
                formatted_value = str(value)
            self.config(key, formatted_value)
        self.blank()


# Global logger instance
_logger = Logger()


def get_logger() -> Logger:
    """Get the global logger instance."""
    return _logger


def set_log_level(level: str):
    """
    Set the global logging level.

    Parameters
    ----------
    level : str
        One of: 'debug', 'info', 'warning', 'error', 'silent'.
    """
    _logger.level = level


def progress(
    iterable,
    desc: str = "Processing",
    unit: str = "it",
    disable: bool = False,
    total: Optional[int] = None,
    log_mode: Optional[bool] = None,
) -> Iterator:
    """
    Convenience function for progress tracking.

    Parameters
    ----------
    iterable : iterable
        Items to iterate over.
    desc : str
        Description of the operation.
    unit : str
        Unit name.
    disable : bool
        Suppress output.
    total : int, optional
        Total count.
    log_mode : bool, optional
        Override global log_mode setting.

    Returns
    -------
    Iterator
        Progress-tracked iterator.

    Examples
    --------
    >>> from bempp_audio.progress import progress
    >>> for freq in progress(frequencies, desc="Frequency sweep", unit="freq"):
    ...     result = solve(freq)
    """
    return ProgressTracker.iterate(iterable, desc, unit, disable, total, log_mode)


def get_device_info() -> Dict[str, Any]:
    """
    Get information about the compute devices available for BEM.

    Returns
    -------
    dict
        Dictionary with device configuration info.
    """
    info = {
        "interface": "unknown",
        "device_type": "unknown",
        "opencl_available": False,
        "gpu_available": False,
        "devices": [],
    }

    try:
        import bempp_cl.api as bempp

        info["interface"] = getattr(bempp, "DEFAULT_DEVICE_INTERFACE", "unknown")
        info["device_type"] = getattr(bempp, "BOUNDARY_OPERATOR_DEVICE_TYPE", "unknown")
        info["opencl_available"] = getattr(bempp, "CPU_OPENCL_DRIVER_FOUND", False) or \
                                   getattr(bempp, "GPU_OPENCL_DRIVER_FOUND", False)
        info["gpu_available"] = getattr(bempp, "GPU_OPENCL_DRIVER_FOUND", False)

        # Try to get OpenCL device details
        try:
            import pyopencl as cl
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    info["devices"].append({
                        "name": device.name,
                        "type": cl.device_type.to_string(device.type),
                        "compute_units": device.max_compute_units,
                        "memory_gb": device.global_mem_size / (1024**3),
                    })
        except ImportError:
            pass
        except Exception:
            pass

        # If no OpenCL, report CPU info
        if not info["devices"]:
            cpu_count = os.cpu_count() or 1
            info["devices"].append({
                "name": "CPU (via Numba)",
                "type": "CPU",
                "compute_units": cpu_count,
                "memory_gb": None,
            })

    except ImportError:
        pass

    return info


def log_device_info():
    """Log the current compute device configuration."""
    logger = get_logger()
    info = get_device_info()

    logger.info(f"Compute backend: {info['interface']}")
    logger.info(f"Device type: {info['device_type']}")

    if info["opencl_available"]:
        logger.info("OpenCL: available")
    else:
        logger.info("OpenCL: not available (using Numba CPU)")

    for device in info["devices"]:
        mem_str = f", {device['memory_gb']:.1f} GB" if device.get('memory_gb') else ""
        logger.info(f"  Device: {device['name']} ({device['compute_units']} compute units{mem_str})")
