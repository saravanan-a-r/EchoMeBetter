"""
GPU readings from NVML, averaged over each logging window.

Why a background thread
-----------------------
One reading taken when a log record arrives would describe the moment
between two steps — the host doing Python work, the GPU briefly idle — and
under-report utilization every time. So a daemon thread samples on a fixed
interval and the monitor collects the window's mean (and the SM clock's
minimum, which is what shows throttling) with each record.

Cost: on the A100X-40C vGPU one sample is four NVML calls, ~2 ms, made
through ctypes with the GIL released; at the default 5 s interval that is
~0.04% of one CPU core, and nothing on the GPU.

Only what the device reports
----------------------------
Each reading is tried once at start and dropped if NVML says it is not
supported. On this vGPU temperature and power are not exposed, so those
charts simply do not appear; on bare metal they do. A reading that fails
later (a driver reset) stops the thread and the monitor reports it once.
"""

from __future__ import annotations

import threading
from typing import Callable

import pynvml
import torch

_GIB = 2**30


class GpuSampler:
    """Sample one GPU on a background thread; `drain()` returns the window's summary."""

    def __init__(self, nvml_uuid: str, *, interval_seconds: float = 5.0) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByUUID(nvml_uuid)
        self._readings = _supported_readings(self._handle)
        self._samples: dict[str, list[float]] = {name: [] for name in self._readings}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, args=(interval_seconds,), name="gpu-sampler", daemon=True
        )
        self._thread.start()

    @classmethod
    def for_device(cls, device: torch.device | str, **kwargs) -> "GpuSampler":
        """The sampler for the CUDA device training runs on, matched by UUID."""
        uuid = torch.cuda.get_device_properties(torch.device(device)).uuid
        return cls(f"GPU-{uuid}", **kwargs)

    def drain(self) -> dict[str, float]:
        """Summary of the samples since the last call, under `gpu/` tags."""
        if self._error is not None:
            raise self._error
        with self._lock:
            samples = {name: values for name, values in self._samples.items() if values}
            self._samples = {name: [] for name in self._readings}
        scalars: dict[str, float] = {}
        for name, values in samples.items():
            mean = sum(values) / len(values)
            if name == "sm_clock_mhz":
                scalars["gpu/sm_clock_mhz"] = mean
                scalars["gpu/sm_clock_min_mhz"] = min(values)
            elif name in ("memory_used_gib", "temperature_c"):
                scalars[f"gpu/{name}"] = max(values)
            else:
                scalars[f"gpu/{name}"] = mean
        return scalars

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    def _run(self, interval_seconds: float) -> None:
        while not self._stop.wait(interval_seconds):
            try:
                reading = {name: read() for name, read in self._readings.items()}
            except Exception as exc:  # reported by the next drain()
                self._error = exc
                return
            with self._lock:
                for name, value in reading.items():
                    self._samples[name].append(value)


def _supported_readings(handle) -> dict[str, Callable[[], float]]:
    """Every reading this device answers, by the name it is charted under."""
    candidates: dict[str, Callable[[], float]] = {
        "utilization_pct": lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
        "memory_bandwidth_pct": lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).memory),
        "sm_clock_mhz": lambda: float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
        "memory_used_gib": lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used / _GIB,
        "temperature_c": lambda: float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        ),
        "power_w": lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
    }
    supported = {}
    for name, read in candidates.items():
        try:
            read()
        except pynvml.NVMLError:
            continue
        supported[name] = read
    return supported
