"""
The device plans and the guards that protect the pretraining run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import resources
from src.config import ResourceConfig
from src.errors import ResourceError


def _cpu(**changes):
    values = dict(cores="0-1", threads=2, nice=19, idle_io=False, precision="fp32")
    values.update(changes)
    return ResourceConfig(**values)


def test_parse_cores():
    assert resources.parse_cores("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)
    assert resources.parse_cores("5, 3 ,3") == (3, 5)
    for bad in ("3-1", "a-b", "", "-2"):
        with pytest.raises(ResourceError):
            resources.parse_cores(bad)


def test_more_threads_than_pinned_cores_is_refused():
    with pytest.raises(ResourceError, match="oversubscribed"):
        resources.plan_devices("cpu", _cpu(cores="0-1", threads=4))


def test_cores_beyond_the_machine_are_refused():
    with pytest.raises(ResourceError, match="exceed"):
        resources.plan_devices("cpu", _cpu(cores="0-100000", threads=1))


def test_cpu_plan_refuses_to_run_with_cuda_visible(monkeypatch):
    """sft.py hides CUDA before torch loads; anything else must not reach the GPU."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    plan = resources.plan_devices("cpu", _cpu())
    with pytest.raises(ResourceError, match="CUDA_VISIBLE_DEVICES"):
        resources.apply_plan(plan)


def test_gpu_plan_refuses_a_gpu_another_process_is_using(monkeypatch):
    monkeypatch.setattr(resources.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(resources, "other_gpu_processes", lambda: [(844164, 34983)])
    plan = resources.plan_devices("gpu", ResourceConfig(cores=None, threads=None, nice=0, idle_io=False, precision="bf16"))
    with pytest.raises(ResourceError, match="844164"):
        resources.apply_plan(plan, require_exclusive_gpu=True)


_THREAD_PROBE = """
import os, sys, threading
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, sys.argv[1])
started = threading.Event(); stop = threading.Event()
def idle():
    started.set(); stop.wait()
thread = threading.Thread(target=idle, daemon=True); thread.start(); started.wait()   # exists BEFORE the plan
from src.config import ResourceConfig
from src import resources
cores = [int(c) for c in sys.argv[2].split(",")]
plan = resources.plan_devices("cpu", ResourceConfig(cores=sys.argv[2], threads=len(cores), nice=19, idle_io=False, precision="fp32"))
resources.apply_plan(plan)                                                # sizes the torch pool too
late = threading.Thread(target=stop.wait, daemon=True); late.start()      # created AFTER the plan
import torch; torch.ones(512, 512).matmul(torch.ones(512, 512)).sum()     # wake the pool
bad = []
for tid in map(int, os.listdir("/proc/self/task")):
    if os.sched_getaffinity(tid) - set(cores) or os.getpriority(os.PRIO_PROCESS, tid) != 19:
        bad.append(tid)
stop.set()
print("BAD", bad)
"""


def test_the_cpu_plan_confines_every_thread_not_just_the_main_one():
    """
    Regression: affinity and nice are per-thread on Linux. Setting them on the
    main thread left torch's import-time thread pool free to run on
    pretraining's cores. Runs in a subprocess so this test process is untouched.
    """
    cores = ",".join(map(str, sorted(os.sched_getaffinity(0))[-2:]))
    result = subprocess.run(
        [sys.executable, "-c", _THREAD_PROBE, str(Path(__file__).resolve().parents[2]), cores],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "BAD []" in result.stdout, result.stdout


def test_gpu_process_parsing_skips_this_process_and_junk():
    output = "844164, 34983\n4242, 512\n[Not Supported], 1\n\n"
    assert resources.parse_gpu_processes(output, own_pid=4242) == [(844164, 34983)]
