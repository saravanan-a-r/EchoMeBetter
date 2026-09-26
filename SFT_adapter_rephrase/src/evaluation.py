"""
Evaluating an adapter: teacher-forced metrics and free-running generation.

Teacher-forced (cheap, every evaluation, no decoding):
  eval/loss, eval/perplexity    mean target-token NLL -- the most stable
                                trend signal, and what selects the best adapter
  eval/token_accuracy           argmax == label, over target tokens
  eval/eos_accuracy             argmax == EOS at the true end: has the adapter
                                learned *when to stop* (format validity)
  eval/ece, eval/confidence     calibration of the top-1 probability
  eval/loss_by_length/*         short (<=32 input tokens), medium (<=128), long

Generation (on a fixed subset, so evaluations are comparable):
  generation/sari, chrf_reference, chrf_input, length_ratio, shorter_rate,
  span_preservation, novel_word_rate, loop_rate, terminated_rate, empty_rate,
  copy_rate, distinct_2, output_tokens
  quality/acceptable_rate       share passing every bar at once (see config)
  reference/*                   the same metrics for the dataset's own
                                references: the ceiling to read the others
                                against (a reference itself is not perfect
                                on every one of them)
"""

from __future__ import annotations

import math
import statistics
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from . import text_metrics as tm
from .batching import collate
from .codec import TextCodec
from .config import AcceptanceConfig, DecodingConfig
from .data import SFTExample
from .decoding import decode_batch

LENGTH_BUCKETS = (("short", 32), ("medium", 128), ("long", math.inf))


@dataclass
class GenerationRecord:
    source: str
    reference: str
    output: str
    terminated: bool
    output_tokens: int


def _bucket(source_tokens: int) -> str:
    for name, limit in LENGTH_BUCKETS:
        if source_tokens <= limit:
            return name
    return LENGTH_BUCKETS[-1][0]


@torch.no_grad()
def teacher_forced_metrics(
    model,
    examples: Sequence[SFTExample],
    codec: TextCodec,
    *,
    batch_size: int,
    device: torch.device,
    autocast=nullcontext,
    calibration_bins: int = 15,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    eos = codec.frame.eos_id
    nll_sum, tokens, correct, eos_total, eos_correct = 0.0, 0, 0, 0, 0
    bucket_nll: dict[str, float] = {}
    bucket_tokens: dict[str, int] = {}
    bin_count = torch.zeros(calibration_bins, dtype=torch.float64)
    bin_confidence = torch.zeros(calibration_bins, dtype=torch.float64)
    bin_correct = torch.zeros(calibration_bins, dtype=torch.float64)

    ordered = sorted(examples, key=lambda e: len(e.encoder_input_ids) + len(e.target_ids))
    try:
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start:start + batch_size]
            batch = collate(chunk, codec.frame).to(device)
            with autocast():
                output = model(
                    encoder_input_ids=batch.encoder_input_ids,
                    decoder_input_ids=batch.decoder_input_ids,
                    encoder_attention_mask=batch.encoder_attention_mask,
                    decoder_attention_mask=batch.decoder_attention_mask,
                )
            logits = output.logits.float()
            mask = batch.labels != -100
            log_probs = F.log_softmax(logits, dim=-1)
            labels = batch.labels.clamp_min(0)
            token_nll = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            confidence, prediction = log_probs.max(dim=-1)
            confidence = confidence.exp()
            hit = (prediction == labels) & mask

            nll_sum += float(token_nll[mask].sum())
            tokens += int(mask.sum())
            correct += int(hit.sum())

            lengths = mask.sum(dim=1)
            last = (lengths - 1).clamp_min(0)
            rows = torch.arange(len(chunk), device=logits.device)
            eos_total += len(chunk)
            eos_correct += int((prediction[rows, last] == eos).sum())

            per_row_nll = (token_nll * mask).sum(dim=1).tolist()
            for example, row_nll, row_tokens in zip(chunk, per_row_nll, lengths.tolist()):
                name = _bucket(example.source_tokens)
                bucket_nll[name] = bucket_nll.get(name, 0.0) + row_nll
                bucket_tokens[name] = bucket_tokens.get(name, 0) + int(row_tokens)

            conf = confidence[mask].double().cpu()
            acc = hit[mask].double().cpu()
            index = (conf * calibration_bins).long().clamp_(0, calibration_bins - 1)
            bin_count.index_add_(0, index, torch.ones_like(conf))
            bin_confidence.index_add_(0, index, conf)
            bin_correct.index_add_(0, index, acc)
    finally:
        model.train(was_training)

    loss = nll_sum / max(tokens, 1)
    total = float(bin_count.sum())
    ece = float(((bin_correct - bin_confidence).abs()).sum() / total) if total else 0.0
    metrics = {
        "eval/loss": loss,
        "eval/perplexity": math.exp(min(loss, 50.0)),
        "eval/token_accuracy": correct / max(tokens, 1),
        "eval/eos_accuracy": eos_correct / max(eos_total, 1),
        "eval/ece": ece,
        "eval/confidence": float(bin_confidence.sum() / total) if total else 0.0,
        "eval/examples": float(len(examples)),
    }
    for name, _ in LENGTH_BUCKETS:
        if bucket_tokens.get(name):
            metrics[f"eval/loss_by_length/{name}"] = bucket_nll[name] / bucket_tokens[name]
    return metrics


@torch.no_grad()
def generate(
    model,
    examples: Sequence[SFTExample],
    codec: TextCodec,
    decoding: DecodingConfig,
    *,
    batch_size: int,
    device: torch.device,
    autocast=nullcontext,
) -> list[GenerationRecord]:
    """Decode `examples` (in their given order) and return one record each."""
    was_training = model.training
    model.eval()
    order = sorted(range(len(examples)), key=lambda i: len(examples[i].encoder_input_ids))
    records: list[GenerationRecord | None] = [None] * len(examples)
    try:
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            chunk = [examples[i] for i in indices]
            batch = collate(chunk, codec.frame).to(device)
            with autocast():
                result = decode_batch(model, batch.encoder_input_ids, batch.encoder_attention_mask, decoding)
            for index, example, ids, done in zip(indices, chunk, result.sequences, result.terminated):
                records[index] = GenerationRecord(
                    source=example.source_text,
                    reference=example.target_text,
                    output=codec.decode(ids).strip(),
                    terminated=done,
                    output_tokens=len(ids),
                )
    finally:
        model.train(was_training)
    return [r for r in records if r is not None]


def generation_metrics(
    records: Sequence[GenerationRecord],
    acceptance: AcceptanceConfig,
    *,
    prefix: str = "generation",
    use_reference_as_output: bool = False,
) -> dict[str, float]:
    """
    Aggregate text metrics over `records`. With `use_reference_as_output`
    the dataset's references are scored as if they were the outputs, which
    gives the ceiling the adapter's numbers are read against.
    """
    rows = []
    for record in records:
        output = record.reference if use_reference_as_output else record.output
        terminated = True if use_reference_as_output else record.terminated
        preservation = tm.span_preservation(record.source, output)
        chrf_input = tm.sentence_chrf(output, record.source)
        ratio = tm.length_ratio(record.source, output)
        loop = tm.has_loop(output)
        empty = not output.strip()
        acceptable = (
            not empty
            and terminated
            and not loop
            and (preservation is None or preservation == 1.0)
            and ratio <= acceptance.max_length_ratio
            and chrf_input >= acceptance.min_input_chrf
        )
        rows.append(
            {
                "sari": tm.sentence_sari(record.source, output, [record.reference]),
                "chrf_reference": tm.sentence_chrf(output, record.reference),
                "chrf_input": chrf_input,
                "length_ratio": ratio,
                "shorter": ratio < 1.0,
                "preservation": preservation,
                "novel": tm.novel_word_rate(record.source, output),
                "loop": loop,
                "terminated": terminated,
                "empty": empty,
                "copy": output.strip() == record.source.strip(),
                "acceptable": acceptable,
            }
        )
    if not rows:
        return {}

    def mean(key: str) -> float:
        values = [float(r[key]) for r in rows if r[key] is not None]
        return statistics.fmean(values) if values else float("nan")

    outputs = [(r.reference if use_reference_as_output else r.output) for r in records]
    metrics = {
        f"{prefix}/sari": mean("sari"),
        f"{prefix}/chrf_reference": mean("chrf_reference"),
        f"{prefix}/chrf_input": mean("chrf_input"),
        f"{prefix}/length_ratio": mean("length_ratio"),
        f"{prefix}/shorter_rate": mean("shorter"),
        f"{prefix}/span_preservation": mean("preservation"),
        f"{prefix}/novel_word_rate": mean("novel"),
        f"{prefix}/loop_rate": mean("loop"),
        f"{prefix}/terminated_rate": mean("terminated"),
        f"{prefix}/empty_rate": mean("empty"),
        f"{prefix}/copy_rate": mean("copy"),
        f"{prefix}/distinct_2": tm.distinct_n(outputs, 2),
        f"{prefix}/examples": float(len(rows)),
    }
    if not use_reference_as_output:
        metrics[f"{prefix}/output_tokens"] = statistics.fmean(r.output_tokens for r in records)
        metrics["quality/acceptable_rate"] = mean("acceptable")
    else:
        metrics[f"{prefix}/acceptable_rate"] = mean("acceptable")
    return metrics


def distributions(records: Sequence[GenerationRecord]) -> dict[str, list[float]]:
    """Per-example values for TensorBoard histograms (the tails matter more than means)."""
    return {
        "generation/length_ratio": [tm.length_ratio(r.source, r.output) for r in records],
        "generation/chrf_input": [tm.sentence_chrf(r.output, r.source) for r in records],
        "generation/sari": [tm.sentence_sari(r.source, r.output, [r.reference]) for r in records],
    }


def samples_markdown(records: Sequence[GenerationRecord], limit: int) -> str:
    """A markdown table of the first `limit` records, for TensorBoard's text tab."""

    def cell(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")[:600]

    lines = ["| # | input | reference | adapter output | stopped |", "|---|---|---|---|---|"]
    for index, record in enumerate(records[:limit]):
        lines.append(
            f"| {index} | {cell(record.source)} | {cell(record.reference)} | "
            f"{cell(record.output)} | {'yes' if record.terminated else '**no**'} |"
        )
    return "\n".join(lines)
