"""
A local TensorBoard dashboard for a training run.

Scope
-----
This package watches a run; it never steers one. It reads the records
`training/`'s `Trainer` already hands to `on_log`, the optimizer through
PyTorch's official step hooks, the GPU through NVML, and the model through its
public `encode` and a caller-supplied `generate` — and writes all of it to
TensorBoard event files. It imports none of this project's other packages, so
it can be dropped into any run that speaks the same record contract.

Plugging it in
--------------
    monitor = TrainingMonitor(
        "runs/pretrain/tensorboard",
        max_steps=config.max_steps,
        start_step=trainer.state.global_step,      # after any resume
        optimizer_stats=OptimizerStats(trainer.optimizer),
        gpu=GpuSampler.for_device(trainer.device),
        probes=ModelProbes(trainer.model, generate=..., decode=..., ...),
        probe_every=config.quick_eval_steps,
    )
    trainer.on_log = monitor                      # or fan out with other sinks
    ...
    monitor.close()

    tensorboard --logdir runs/pretrain/tensorboard

Every component is optional; a monitor with none of them still charts
everything in the trainer's records.

Every metric carries a plain-English title and explanation from
`catalogue.py`, which also lays out the Custom Scalars tab and writes the
metric guide to the Text tab. A new metric needs an entry there; the
end-to-end dashboard test fails on one that has none.

Two guarantees
--------------
**It cannot end the run.** Every component runs inside the training loop, so
each is guarded: the first error disables that component, says so once in the
run's output, and training carries on without it. A dashboard is worth far
less than a multi-week run.

**It does not change the run.** Nothing here writes to a parameter, a
gradient, the optimizer's state or a random number generator — the model
probes run under `no_grad` in eval mode with their own seeded `Random`, and
put the model back in the mode they found it. `test_optimizer_stats.py`
asserts a monitored run and an unmonitored one end bit-identical.

What it costs is measured per component in the module docstrings; the
expensive ones run on a cadence, never on every step.
"""

from .dashboard import TrainingMonitor
from .gpu import GpuSampler
from .optimizer_stats import OptimizerStats
from .probes import ModelProbes, ProbeExample, probe_examples

__all__ = [
    "TrainingMonitor",
    "OptimizerStats",
    "GpuSampler",
    "ModelProbes",
    "ProbeExample",
    "probe_examples",
]
