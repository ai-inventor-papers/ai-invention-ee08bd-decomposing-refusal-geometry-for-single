"""Hardware detection and resource limits (cgroup-aware, per aii-use-hardware)."""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

from loguru import logger

CONTAINER_RAM_LIMIT_BYTES = 14_000_000_000  # measured: cgroup v1 memory.limit_in_bytes
RAM_BUDGET_BYTES = 12_500_000_000           # abort threshold (RSS watchdog), leaves headroom
RAM_HARD_ABORT_BYTES = 13_200_000_000
NUM_CPUS = 2


def detect_cpus() -> int:
    """Detect actual CPU allocation (cgroup quota -> affinity -> os.cpu_count)."""
    try:  # cgroups v1 quota
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1


def container_ram_gb() -> float:
    for p in ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory.max"):
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            continue
    return 14.0


def current_rss_gb() -> float:
    """True anonymous RSS from the cgroup (page cache is reclaimable and must NOT count)."""
    try:
        stat = Path("/sys/fs/cgroup/memory/memory.stat").read_text().split()
        d = dict(zip(stat[0::2], map(int, stat[1::2])))
        # anonymous + kernel stack/swap-backed: what the OOM killer actually counts
        return (d.get("anon", 0) + d.get("kernel_stack", 0) + d.get("sock", 0)) / 1e9
    except (FileNotFoundError, ValueError):
        pass
    try:
        return int(Path("/proc/self/status").read_text().split("VmRSS:")[1].split()[0]) * 1024 / 1e9
    except (IndexError, ValueError, FileNotFoundError):
        return 0.0


class RSSWatchdog:
    """Poll cgroup RSS; abort before the container OOM-killer strikes.

    NOTE: we deliberately do NOT set RLIMIT_AS on torch CPU workloads - PyTorch's
    threadpools reserve tens of GB of *virtual* address space, so an RLIMIT_AS cap
    trips spuriously long before RSS approaches the cgroup limit. We cap RSS itself
    instead (the resource that actually kills the container).
    """

    def __init__(self, limit_bytes: float = RAM_HARD_ABORT_BYTES, poll_s: float = 5.0):
        self.limit = limit_bytes
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss = current_rss_gb()
            if rss * 1e9 > self.limit:
                logger.error(f"RSS watchdog: {rss:.2f} GB exceeds hard limit {self.limit/1e9:.2f} GB - aborting")
                os._exit(137)  # fail fast and visibly, same as an OOM kill would
            self._stop.wait(self.poll_s)

    def __enter__(self) -> "RSSWatchdog":
        self._thread = threading.Thread(target=self._loop, daemon=True, name="rss-watchdog")
        self._thread.start()
        logger.info(f"RSS watchdog on: abort at {self.limit/1e9:.1f} GB (cgroup limit {container_ram_gb():.1f} GB)")
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()


def configure_threads() -> None:
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(NUM_CPUS))
    os.environ.setdefault("MKL_NUM_THREADS", str(NUM_CPUS))
    try:
        import torch

        torch.set_num_threads(NUM_CPUS)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def log_hardware() -> None:
    import torch

    logger.info(
        f"HW: cpus={detect_cpus()} ram_limit={container_ram_gb():.1f}GB "
        f"cuda={torch.cuda.is_available()} threads={torch.get_num_threads()}"
    )
