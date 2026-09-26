"""
Run logging: a text log, a machine-readable metrics file, and TensorBoard.

    SFT_adapter_rephrase/logs/<run>/sft.log          human log (also on stdout)
    SFT_adapter_rephrase/logs/<run>/metrics.jsonl    one JSON object per logged step
    SFT_adapter_rephrase/logs/<run>/tensorboard/     event files

The TensorBoard directory is this module's own, never `runs/pretrain/
tensorboard` (the config refuses anything under `runs/pretrain`). Serve it
on its own port with `SFT_adapter_rephrase/tensorboard.sh` -- pretraining's
TensorBoard keeps 6006.

`metrics.jsonl` duplicates the scalars on purpose: it can be read with
`json` alone, which TensorBoard's event files cannot.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

LOGGER_NAME = "sft_adapter_rephrase"


def setup_logger(log_file: Path, *, to_stdout: bool = True) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if to_stdout:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        logger.addHandler(stream)
    return logger


class MetricsWriter:
    """TensorBoard scalars/text/histograms plus the JSONL mirror of the scalars."""

    def __init__(self, log_dir: Path, *, tensorboard: bool = True) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = log_dir / "metrics.jsonl"
        self._jsonl = self.jsonl_path.open("a", encoding="utf-8")
        self._tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    def scalars(self, step: int, values: Mapping[str, float]) -> None:
        clean = {k: float(v) for k, v in values.items() if v is not None and math.isfinite(float(v))}
        if not clean:
            return
        self._jsonl.write(json.dumps({"step": step, "time": time.time(), **clean}) + "\n")
        self._jsonl.flush()
        if self._tb is not None:
            for key, value in clean.items():
                self._tb.add_scalar(key, value, step)

    def text(self, step: int, tag: str, body: str) -> None:
        if self._tb is not None:
            self._tb.add_text(tag, body, step)

    def histogram(self, step: int, tag: str, values: Sequence[float]) -> None:
        if self._tb is not None and values:
            import numpy as np

            self._tb.add_histogram(tag, np.asarray(values, dtype=np.float64), step)

    def flush(self) -> None:
        self._jsonl.flush()
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        self.flush()
        self._jsonl.close()
        if self._tb is not None:
            self._tb.close()
