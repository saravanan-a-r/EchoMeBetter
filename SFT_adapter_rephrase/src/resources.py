"""
Device plans: where the run executes, and the guarantees that keep it from
disturbing the pretraining run on the same machine.

CPU plan (`--device cpu`), for running *beside* pretraining:
  - CUDA hidden from the process (`CUDA_VISIBLE_DEVICES=""`, set by `sft.py`
    before torch loads), so nothing can create a context on the busy GPU;
  - pinned to the configured cores (default 0-21: NUMA nodes 0-1, away from
    pretraining's threads on node 2 with the GPU) -- every thread, including
    the pool torch starts at import, and verified afterwards;
  - `threads` intra-op threads, never more than the pinned cores (OpenMP
    oversubscription slows everything several-fold);
  - `nice 19` and the idle I/O class, so pretraining's corpus streaming
    always wins the disk.

GPU plan (`--device gpu`), for after pretraining:
  - refuses to start while any other process holds the GPU, unless
    explicitly allowed -- an accidental `--device gpu` during pretraining
    would otherwise allocate next to a job using 35 of 40 GB;
  - all cores, bf16 autocast.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial

import torch

from .config import ResourceConfig
from .errors import ResourceError


@dataclass(frozen=True)
class DevicePlan:
    device: torch.device
    precision: str
    cores: tuple[int, ...] | None
    threads: int
    nice: int
    idle_io: bool

    def autocast(self):
        """A zero-argument context factory for the forward passes."""
        if self.precision == "bf16":
            return partial(torch.autocast, device_type=self.device.type, dtype=torch.bfloat16)
        return nullcontext

    def describe(self) -> str:
        cores = "all" if self.cores is None else _format_cores(self.cores)
        return (
            f"device={self.device} precision={self.precision} cores={cores} "
            f"threads={self.threads} nice={self.nice} idle_io={self.idle_io}"
        )


def parse_cores(spec: str) -> tuple[int, ...]:
    """'0-21' / '0-3,8,10-11' -> sorted unique core ids."""
    cores: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                low, high = (int(x) for x in part.split("-", 1))
                if low > high:
                    raise ValueError
                cores.update(range(low, high + 1))
            else:
                cores.add(int(part))
        except ValueError:
            raise ResourceError(f"cannot parse core list {spec!r} (expected e.g. '0-21' or '0-3,8')") from None
    if not cores or min(cores) < 0:
        raise ResourceError(f"core list {spec!r} is empty or negative")
    return tuple(sorted(cores))


def plan_devices(device: str, resources: ResourceConfig) -> DevicePlan:
    """Resolve the config into a concrete plan; validates, applies nothing."""
    available = os.cpu_count() or 1
    if resources.cores is None:
        cores = None
        core_count = available
    else:
        cores = parse_cores(resources.cores)
        if max(cores) >= available:
            raise ResourceError(f"cores {resources.cores!r} exceed this machine's {available} CPUs")
        core_count = len(cores)
    threads = resources.threads or core_count
    if threads > core_count:
        raise ResourceError(
            f"threads={threads} exceeds the {core_count} pinned cores; oversubscribed OpenMP "
            f"threads slow down several-fold"
        )
    if device == "cpu":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device("cuda")
    return DevicePlan(
        device=torch_device,
        precision=resources.precision,
        cores=cores,
        threads=threads,
        nice=resources.nice,
        idle_io=resources.idle_io,
    )


def apply_plan(plan: DevicePlan, *, require_exclusive_gpu: bool = False) -> list[str]:
    """
    Apply the plan to this process. Returns human-readable notes for the log
    (best-effort steps that could not be applied are noted, not fatal; the
    guarantees that matter -- CUDA hidden, core pinning, the GPU check -- raise).
    """
    notes: list[str] = []
    if plan.device.type == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ResourceError(
                "CPU plan requires CUDA_VISIBLE_DEVICES='' before torch initializes CUDA; "
                "launch through sft.py, which sets it"
            )
    else:
        if not torch.cuda.is_available():
            raise ResourceError("--device gpu was requested but CUDA is not available")
        if require_exclusive_gpu:
            others = other_gpu_processes()
            if others:
                listed = ", ".join(f"pid {pid} ({mib} MiB)" for pid, mib in others)
                raise ResourceError(
                    f"the GPU is in use by {listed}. SFT on the GPU is meant for after "
                    f"pretraining; pass --allow-shared-gpu only if you are sure"
                )

    # Affinity, nice and I/O priority are per *thread* on Linux, and importing
    # torch has already started a thread pool (one thread per core, 42 here);
    # resizing it below starts another. Setting them on the main thread alone
    # left those threads free to run on every core -- pretraining's included
    # -- at the default priority. So the pool is sized first, then every
    # existing thread is set, then every thread is verified; threads created
    # later inherit from their (already set) creator.
    torch.set_num_threads(plan.threads)
    threads = _thread_ids()
    for tid in threads:
        if plan.cores is not None:
            _for_live_thread(os.sched_setaffinity, tid, plan.cores)
        if plan.nice > 0 and _nice(tid) < plan.nice:
            _for_live_thread(os.setpriority, os.PRIO_PROCESS, tid, plan.nice)
    if plan.cores is not None:
        notes.append(f"pinned {len(threads)} threads to cores {_format_cores(plan.cores)}")
    if plan.nice > 0:
        notes.append(f"nice {plan.nice}")

    if plan.idle_io:
        try:
            subprocess.run(["ionice", "-c", "3", "-p", *map(str, threads)], check=True, capture_output=True)
            notes.append("idle I/O class")
        except (OSError, subprocess.CalledProcessError) as exc:
            notes.append(f"could not set idle I/O class ({exc}); continuing")

    live = _thread_ids()
    if plan.cores is not None:
        escaped = [tid for tid in live if not _affinity(tid) <= set(plan.cores)]
        if escaped:
            raise ResourceError(f"threads {escaped} are not confined to cores {_format_cores(plan.cores)}")
    unniced = [tid for tid in live if _nice(tid) is not None and _nice(tid) < plan.nice]
    if unniced:
        raise ResourceError(f"threads {unniced} run below nice {plan.nice}")
    return notes


def _thread_ids() -> list[int]:
    return sorted(int(entry) for entry in os.listdir("/proc/self/task"))


def _nice(tid: int) -> int | None:
    try:
        return os.getpriority(os.PRIO_PROCESS, tid)
    except ProcessLookupError:
        return None


def _affinity(tid: int) -> set[int]:
    try:
        return os.sched_getaffinity(tid)
    except ProcessLookupError:  # exited since it was listed
        return set()


def _for_live_thread(function, *args) -> None:
    """Apply `function`, ignoring a thread that exited since it was listed."""
    try:
        function(*args)
    except ProcessLookupError:
        pass


def other_gpu_processes() -> list[tuple[int, int]]:
    """(pid, MiB) of every other process with a GPU context, via nvidia-smi (no CUDA init)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ResourceError(f"cannot query GPU processes with nvidia-smi: {exc}") from exc
    return parse_gpu_processes(result.stdout, own_pid=os.getpid())


def parse_gpu_processes(output: str, own_pid: int) -> list[tuple[int, int]]:
    processes = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        memory = int(parts[1]) if parts[1].isdigit() else 0
        if pid != own_pid:
            processes.append((pid, memory))
    return processes


def memory_stats(device: torch.device) -> dict[str, float]:
    stats = {}
    try:
        with open("/proc/self/statm") as handle:
            resident_pages = int(handle.read().split()[1])
        stats["perf/rss_gb"] = resident_pages * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (OSError, ValueError, IndexError):
        pass
    if device.type == "cuda":
        stats["perf/gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    return stats


def _format_cores(cores: tuple[int, ...]) -> str:
    ranges, start, previous = [], cores[0], cores[0]
    for core in cores[1:]:
        if core != previous + 1:
            ranges.append(f"{start}-{previous}" if start != previous else str(start))
            start = core
        previous = core
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)
