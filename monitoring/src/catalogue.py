"""
What every dashboard metric means, in one place.

The catalogue is the dashboard's plain-English layer. From it come:

  - each scalar's title and explanation, attached to its first data point,
    which the Scalars tab shows as the card title and an (i) button;
  - the Custom Scalars tab: charts with readable titles, grouped by the
    question they answer, the overview first;
  - `guide`, the metric reference in the Text tab, whose link also opens the
    Time Series tab with the overview charts pinned on top.

TensorBoard reads a tag's title and explanation from the first data point
the run wrote for it and ignores them after that. A run that started before
a metric had an entry here keeps its bare tag name for that metric.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import quote

from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.plugins.scalar.metadata import create_summary_metadata
from torch.utils.tensorboard.summary import scalar


@dataclass(frozen=True)
class Metric:
    """One tag, or one family of tags, and what to read into it."""

    pattern: str  # full match over the tag; named groups fill `title`
    title: str
    meaning: str
    healthy: str
    act: str

    def description(self) -> str:
        return f"{self.meaning}\n\n**Healthy:** {self.healthy}\n\n**Act when:** {self.act}"


@dataclass(frozen=True)
class Section:
    title: str
    intro: str
    metrics: tuple[Metric, ...]
    # Custom Scalars charts: title -> tag patterns overlaid on that chart.
    charts: dict[str, list[str]]


_GROUPS = (
    "Groups: `decayed` (weight matrices), `no_decay` (norms, embeddings), "
    "`position_bias` (relative-position tables), `query` (attention query projections)."
)

SECTIONS: tuple[Section, ...] = (
    Section(
        "Overview",
        "The six charts that say whether the run is healthy. Everything below explains them.",
        (),
        {
            "Loss: training vs held-out": [r"^loss/train$", r"^eval/[^/]+/loss$"],
            "Loss spikes (loss / recent median; alert at 1.2)": [r"^loss/spike_ratio$"],
            "Gradient norm before clipping (clipped to 1.0)": [r"^optim/grad_norm$"],
            "Learning rate": [r"^optim/learning_rate$"],
            "Encoder word-order sensitivity (lower is better)": [r"^position/shuffle_cosine$"],
            "Time remaining (days)": [r"^throughput/eta_days$"],
        },
    ),
    Section(
        "Loss",
        "Cross-entropy per graded token. Training values are logged every step; held-out "
        "values at each evaluation (quick every 1,000 steps, master on its own cadence).",
        (
            Metric(
                r"loss/train",
                "Training loss",
                "Mean cross-entropy per graded token over this step's batch.",
                "Falls steeply through warmup, then keeps falling slowly with step-to-step noise.",
                "It flattens for thousands of steps early on, climbs for hundreds of steps, "
                "or becomes NaN.",
            ),
            Metric(
                r"loss/perplexity",
                "Training perplexity",
                "exp(training loss): how many tokens the model is effectively choosing between.",
                "Starts in the tens of thousands and falls with the loss.",
                "Same as the training loss: read the loss, this is its readable form.",
            ),
            Metric(
                r"loss/train_(?P<mode>[^/]+)",
                "Training loss, {mode} mode",
                "Training loss over this step's rows of one UL2 denoiser: R (short spans), "
                "X (extreme spans) or S (prefix to continuation). Modes differ in "
                "difficulty, so compare each with itself.",
                "Every mode falls.",
                "One mode flattens while the others fall: the model is neglecting that task.",
            ),
            Metric(
                r"loss/spike_ratio",
                "Loss spike ratio",
                "This step's loss divided by the median of the previous 100 steps. The "
                "console prints a WARNING on the step a ratio first crosses 1.2.",
                "Close to 1.0, within about 0.9 to 1.1.",
                "It crosses 1.2 repeatedly, or stays high for more than a few dozen steps: "
                "check the gradient norm and the batch at that step.",
            ),
            Metric(
                r"eval/(?P<tier>[^/]+)/loss",
                "Held-out loss ({tier})",
                "Mean cross-entropy on the fixed {tier} evaluation set, which the model "
                "never trains on.",
                "Falls with the training loss and stays close to it; this is the number "
                "that picks the best checkpoint.",
                "It rises while the training loss falls (overfitting), or the gap to the "
                "training loss widens quickly.",
            ),
            Metric(
                r"eval/(?P<tier>[^/]+)/perplexity",
                "Held-out perplexity ({tier})",
                "exp(held-out loss) on the {tier} evaluation set.",
                "Tracks the held-out loss.",
                "Read the held-out loss; this is its readable form.",
            ),
            Metric(
                r"eval/(?P<tier>[^/]+)/loss_(?P<mode>[^/]+)",
                "Held-out loss ({tier}), {mode} mode",
                "Held-out loss over the {tier} set's rows of one UL2 denoiser.",
                "Every mode falls from one evaluation to the next.",
                "One mode stops improving or gets worse between evaluations.",
            ),
            Metric(
                r"eval/best_metric",
                "Best held-out loss so far",
                "The lowest held-out loss of the checkpoint-selecting tier up to this step.",
                "Steps down at most evaluations.",
                "It stays flat across several evaluations: the run has stopped improving.",
            ),
            Metric(
                r"eval/best_step",
                "Step of the best checkpoint",
                "The step whose checkpoint currently holds the best held-out loss; that "
                "checkpoint is kept however many others are pruned.",
                "Moves forward with each evaluation.",
                "It stays put while training continues: later checkpoints are worse.",
            ),
        ),
        {
            "Training loss by UL2 mode": [r"^loss/train_"],
            "Held-out loss by UL2 mode": [r"^eval/[^/]+/loss_"],
            "Perplexity: training vs held-out": [r"^loss/perplexity$", r"^eval/[^/]+/perplexity$"],
            "Best held-out loss so far": [r"^eval/best_metric$"],
            "Step of the best checkpoint": [r"^eval/best_step$"],
        },
    ),
    Section(
        "Optimizer",
        "What each optimizer step did. The overall numbers are logged every step; the "
        "per-group ones are sampled every 10 steps. " + _GROUPS + " Read these charts on "
        "a log scale.",
        (
            Metric(
                r"optim/learning_rate",
                "Learning rate",
                "The base learning rate at this step. Groups with a multiplier run at a "
                "multiple of it (see Learning rate in force).",
                "Rises linearly through warmup (the first 1% of steps), then follows the "
                "schedule.",
                "Its shape differs from the configured schedule.",
            ),
            Metric(
                r"optim/grad_norm",
                "Gradient norm before clipping",
                "Global L2 norm of all gradients before clipping to 1.0. Above 1.0, the "
                "step is scaled down to 1.0.",
                "Higher and noisy in the first steps, then settling into a stable band.",
                "A sustained climb, or jumps of 10x: these usually come just before a loss spike.",
            ),
            Metric(
                r"optim/skipped_steps",
                "Skipped steps (running total)",
                "Steps thrown away because a gradient was NaN or infinite.",
                "0, or a rare single step.",
                "It grows steadily: numerics are unstable. Check the loss and gradient norm "
                "around the first skip.",
            ),
            Metric(
                r"update_ratio/(?P<group>[^/]+)",
                "Update size / weight size, {group}",
                "||w_after - w_before|| / ||w_before|| for one optimizer step: how far one "
                "step moves this group, relative to its size. The most direct check that a "
                "group is learning at a sensible rate.",
                "Around 1e-3 for weight matrices (1e-4 to 1e-2 is normal); groups with an "
                "lr multiplier sit proportionally higher or lower.",
                "A group is 10x above its usual level (being thrashed), or near 0 (frozen). "
                "The first run's position-blind encoder showed up here as a frozen "
                "position_bias group.",
            ),
            Metric(
                r"weight_rms/(?P<group>[^/]+)",
                "Weight RMS, {group}",
                "Root mean square of the group's weights.",
                "Drifts slowly. Weight decay holds matrix groups roughly level.",
                "It grows without bound, or a group that should learn does not move at all.",
            ),
            Metric(
                r"grad_norm/(?P<group>[^/]+)",
                "Gradient norm after clipping, {group}",
                "L2 norm of this group's gradients as the optimizer received them. "
                "Clipping scales every group by the same factor, so the groups compare "
                "exactly as before clipping.",
                "Stable relative to the other groups.",
                "One group's share jumps or collapses compared with the others.",
            ),
            Metric(
                r"group_lr/(?P<group>[^/]+)",
                "Learning rate in force, {group}",
                "The learning rate the optimizer applied to this group: the base rate "
                "times the group's multiplier.",
                "Parallel to the base learning rate, offset by the group's multiplier.",
                "A group sits on the wrong line: its multiplier did not reach the schedule.",
            ),
        ),
        {
            "Update size / weight size, per group": [r"^update_ratio/"],
            "Weight RMS, per group": [r"^weight_rms/"],
            "Gradient norm after clipping, per group": [r"^grad_norm/"],
            "Learning rate in force, per group": [r"^group_lr/"],
            "Skipped steps (running total)": [r"^optim/skipped_steps$"],
        },
    ),
    Section(
        "Position awareness",
        "Whether the encoder uses word order: the failure that ended the first "
        "pretraining run. Measured by probes at every quick evaluation.",
        (
            Metric(
                r"position/shuffle_cosine",
                "Encoder word-order sensitivity",
                "Shuffle the words of held-out inputs and compare each word's encoding "
                "before and after (cosine similarity). An encoder that uses word order "
                "encodes a moved word differently.",
                "Falls from about 1.0 at initialization towards 0.5 (a healthy T5).",
                "It stays above about 0.95 after the first few thousand steps: the encoder is "
                "position-blind. The first run measured 0.998 at step 5,111.",
            ),
            Metric(
                r"position/(?P<table>.+)_absmax",
                "Position-bias table, largest entry ({table})",
                "The largest absolute value in a relative-position-bias table. The table "
                "has to reach a few units before it shapes attention.",
                "Grows over the first thousands of steps.",
                "It stays near its initial value: the other sign of a position-blind model.",
            ),
        ),
        {
            "Encoder word-order sensitivity (lower is better)": [r"^position/shuffle_cosine$"],
            "Position-bias tables, largest entry": [r"^position/.*_absmax$"],
        },
    ),
    Section(
        "Speed",
        "Pace measured between log records, so evaluation, probes and checkpoint saves "
        "are included.",
        (
            Metric(
                r"throughput/seconds_per_step",
                "Seconds per step",
                "Wall-clock time between two training records.",
                "A flat band, with a tall bar at each evaluation and checkpoint save.",
                "The band steps up and stays there: the GPU is throttling, or the host or "
                "data loading has slowed.",
            ),
            Metric(
                r"throughput/tokens_per_second",
                "Tokens per second",
                "Input plus graded tokens processed per second.",
                "A flat band.",
                "It drops and stays down. Check the GPU's utilization and SM clock.",
            ),
            Metric(
                r"throughput/eta_days",
                "Time remaining (days)",
                "Steps left times this session's average time per step.",
                "Falls in a straight line.",
                "It flattens or rises: the run has slowed.",
            ),
            Metric(
                r"tokens/(?P<kind>graded|input|total)",
                "Tokens seen, {kind}",
                "Running total of tokens the model has processed: `input` (encoder "
                "inputs), `graded` (target tokens the loss is computed on), and `total`.",
                "Straight lines.",
                "A line bends: batches have become shorter or emptier.",
            ),
        ),
        {
            "Tokens per second": [r"^throughput/tokens_per_second$"],
            "Seconds per step": [r"^throughput/seconds_per_step$"],
            "Time remaining (days)": [r"^throughput/eta_days$"],
            "Tokens seen": [r"^tokens/"],
        },
    ),
    Section(
        "Memory and GPU",
        "PyTorch's own memory counters every step, plus the GPU driver's readings "
        "averaged over each step. Temperature and power appear only on GPUs that "
        "report them (the A100X-40C vGPU does not).",
        (
            Metric(
                r"memory/peak_allocated_gib",
                "Peak memory allocated (GiB)",
                "The most GPU memory PyTorch's tensors held during the step.",
                "Flat after the first steps.",
                "It creeps up over time: something is holding onto tensors.",
            ),
            Metric(
                r"memory/peak_reserved_gib",
                "Peak memory reserved (GiB)",
                "The most memory PyTorch's caching allocator held from the driver during "
                "the step, used or not.",
                "Flat, a few GiB above allocated, and clear of the card's capacity.",
                "It approaches the card's capacity: an out-of-memory error is close.",
            ),
            Metric(
                r"memory/fragmentation_gib",
                "Allocator fragmentation (GiB)",
                "Reserved minus allocated: memory the allocator holds but cannot use.",
                "Flat.",
                "It grows over days: the early warning of an out-of-memory error with "
                "memory still nominally free.",
            ),
            Metric(
                r"gpu/utilization_pct",
                "GPU busy (%)",
                "Share of time the GPU was running a kernel, averaged over the step.",
                "High and flat, apart from dips at evaluations and saves.",
                "It drops and stays low: the GPU is waiting on the host or data.",
            ),
            Metric(
                r"gpu/memory_bandwidth_pct",
                "Memory bandwidth busy (%)",
                "Share of time the GPU's memory was being read or written.",
                "A stable band.",
                "It shifts sharply without a config change.",
            ),
            Metric(
                r"gpu/sm_clock_mhz",
                "SM clock, mean (MHz)",
                "The streaming multiprocessors' clock, averaged over the step.",
                "Flat at the card's working clock.",
                "It drops: the GPU is throttling for heat or power.",
            ),
            Metric(
                r"gpu/sm_clock_min_mhz",
                "SM clock, lowest (MHz)",
                "The lowest clock sampled during the step. Short throttles show here "
                "before they move the mean.",
                "Close to the mean.",
                "It dips repeatedly while the mean holds.",
            ),
            Metric(
                r"gpu/memory_used_gib",
                "Device memory used (GiB)",
                "Memory in use on the card as the driver sees it, including memory "
                "outside PyTorch.",
                "Flat, a little above PyTorch's reserved memory.",
                "It grows while PyTorch's reserved memory does not: another process "
                "is using the card.",
            ),
            Metric(
                r"gpu/temperature_c",
                "GPU temperature (C)",
                "Highest temperature sampled during the step.",
                "Stable, well below 85 C.",
                "It trends towards 85 C or more, where clocks throttle.",
            ),
            Metric(
                r"gpu/power_w",
                "GPU power (W)",
                "Mean power draw over the step.",
                "A stable band.",
                "It falls while utilization holds: the card is power-capped.",
            ),
        ),
        {
            "Peak memory: allocated and reserved (GiB)": [r"^memory/peak_"],
            "Allocator fragmentation (GiB)": [r"^memory/fragmentation_gib$"],
            "Device memory used (GiB)": [r"^gpu/memory_used_gib$"],
            "GPU busy and memory bandwidth (%)": [r"^gpu/.*_pct$"],
            "SM clock: mean and lowest (MHz)": [r"^gpu/sm_clock"],
        },
    ),
)

# Time Series cards the guide's overview link pins on top.
OVERVIEW_TAGS = (
    "loss/train",
    "eval/quick/loss",
    "loss/spike_ratio",
    "optim/grad_norm",
    "optim/learning_rate",
    "position/shuffle_cosine",
    "throughput/tokens_per_second",
    "throughput/eta_days",
)

_COMPILED = [
    (re.compile(metric.pattern), metric) for section in SECTIONS for metric in section.metrics
]


def describe(tag: str) -> tuple[str, str] | None:
    """`(title, explanation)` for a tag, or `None` if the catalogue has no entry."""
    for pattern, metric in _COMPILED:
        match = pattern.fullmatch(tag)
        if match:
            fields = match.groupdict()
            return metric.title.format(**fields), metric.description().format(**fields)
    return None


def described_scalar(tag: str, value: float) -> Summary | None:
    """
    A tag's first scalar event, carrying its title and explanation; `None` if
    the catalogue has no entry for the tag.

    In the tensor form the monitor writes every scalar in
    (`add_scalar(..., new_style=True)`): TensorBoard's default data loader
    drops the title and explanation from the older `simple_value` form.
    """
    described = describe(tag)
    if described is None:
        return None
    summary = scalar(tag, value, new_style=True)
    summary.value[0].metadata.CopyFrom(create_summary_metadata(*described))
    return summary


def layout() -> dict[str, dict[str, list]]:
    """The Custom Scalars layout, in `SummaryWriter.add_custom_scalars` form."""
    return {
        section.title: {title: ["Multiline", tags] for title, tags in section.charts.items()}
        for section in SECTIONS
    }


def overview_link() -> str:
    """Relative URL of the Time Series tab with the overview charts pinned."""
    cards = [{"plugin": "scalars", "tag": tag} for tag in OVERVIEW_TAGS]
    pinned = quote(json.dumps(cards, separators=(",", ":")), safe="")
    return f"/?pinnedCards={pinned}#timeseries"


def guide() -> str:
    """The metric reference shown in the Text tab, as Markdown."""
    lines = [
        "## How to read this dashboard",
        "",
        f"**[Open the overview]({overview_link()})**: the Time Series tab with the "
        "health charts pinned on top.",
        "",
        "| Tab | What it is for |",
        "|---|---|",
        "| **Custom Scalars** | The curated view: related curves on one chart, grouped by "
        "the question they answer. Start here. |",
        "| **Time Series** | Every metric, searchable with the filter box. Pin cards to "
        "keep them on top. |",
        "| **Scalars** | Every metric with its plain-English title. The (i) button on a "
        "card explains the metric. |",
        "| **Histograms / Distributions** | Weight and gradient values per parameter "
        "group, at every quick evaluation. |",
        "| **Text** | Greedy samples per UL2 mode (`samples/`), the session log "
        "(`run/sessions`), and this guide. |",
        "",
        "Settings (gear icon, top right): **smoothing** 0.6 to 0.9 makes the loss trend "
        "readable. Switch to a **log scale** for gradient norms and update ratios. The "
        "moon icon toggles dark mode.",
        "",
    ]
    for section in SECTIONS:
        if not section.metrics:
            continue
        lines += [
            f"### {section.title}",
            "",
            section.intro,
            "",
            "| Metric | Tag | What it means | Healthy | Act when |",
            "|---|---|---|---|---|",
        ]
        for metric in section.metrics:
            tag = re.sub(r"\(\?P<(\w+)>[^)]*\)", r"<\1>", metric.pattern)
            generic = {name: f"`{name}`" for name in re.compile(metric.pattern).groupindex}
            lines.append(
                "| "
                + " | ".join(
                    _cell(text.format(**generic))
                    for text in (metric.title, f"`{tag}`", metric.meaning, metric.healthy, metric.act)
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
