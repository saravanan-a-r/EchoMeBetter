"""
Reference-light quality metrics for a rewrite, standard library only.

A rewrite has many correct answers, so exact match and token F1 against the
one reference punish valid outputs. The metrics here each answer one
question, and most need no reference at all:

  SARI (Xu et al. 2016)   the standard rewriting/simplification metric: scores
                          what was ADDED, KEPT and DELETED relative to the
                          INPUT, checked against the reference. A copy of the
                          input scores low; a valid paraphrase that is not the
                          reference still scores partially.
  chrF (Popovic 2015)     character n-gram F-score. Against the reference:
                          closeness in wording. Against the INPUT: how much of
                          the source survived -- the meaning-retention proxy.
  span preservation       numbers, URLs, e-mails, @mentions and `code` from the
                          input must survive verbatim. Deterministic: this is
                          the preservation validator the product design calls
                          for, and the one failure a user will not forgive.
  novel-word rate         share of output words absent from the input -- a
                          hallucination proxy for a task that should reword
                          what is there, not add to it.
  loop                    a word trigram occurring 3+ times: the greedy-loop
                          failure (watch item 4) caught at the text level.
  length ratio            output/input characters. "Concise" means < 1.

SARI follows the HuggingFace `evaluate` implementation (sacreBLEU 13a
tokenization, lowercased, n-grams 1-4, keep/add F1 and deletion precision)
and reproduces its documented example value exactly (tested). One caveat
of that implementation: it defines 0/0 = 1, so an output that deletes
nothing gets full deletion precision and a verbatim copy of the input still
scores high (~88 on a typical concise pair). SARI is therefore always read
beside `copy_rate` and `length_ratio`, which catch exactly that failure.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

# -- tokenization ---------------------------------------------------------

_13A_RULES = (
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
)


def tokenize_13a(text: str) -> str:
    """sacreBLEU's default ('13a', mteval-v13a) tokenizer."""
    text = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in text:
        text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = f" {text} "
    for pattern, replacement in _13A_RULES:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def words(text: str) -> list[str]:
    """Lowercased alphanumeric words, for the word-level metrics."""
    return [w.lower() for w in _WORD.findall(text)]


# -- SARI -------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _sari_ngram(source: list[str], candidate: list[str], references: list[list[str]], num_refs: int):
    reference_counter = Counter(g for ref in references for g in ref)
    source_rep = Counter({g: c * num_refs for g, c in Counter(source).items()})
    candidate_rep = Counter({g: c * num_refs for g, c in Counter(candidate).items()})

    # KEEP
    keep_rep = source_rep & candidate_rep
    keep_good = keep_rep & reference_counter
    keep_all = source_rep & reference_counter
    keep_precision_sum = sum(keep_good[g] / keep_rep[g] for g in keep_rep)
    keep_recall_sum = sum(keep_good[g] for g in keep_rep)
    keep_precision = keep_precision_sum / len(keep_rep) if keep_rep else 1.0
    keep_recall = keep_recall_sum / sum(keep_all.values()) if keep_all else 1.0
    keep = _f1(keep_precision, keep_recall)

    # DELETION (precision only)
    delete_rep = source_rep - candidate_rep
    delete_good = delete_rep - reference_counter
    delete_precision = (
        sum(delete_good[g] / delete_rep[g] for g in delete_rep) / len(delete_rep) if delete_rep else 1.0
    )

    # ADDITION
    added = set(candidate_rep) - set(source_rep)
    added_good = added & set(reference_counter)
    added_all = set(reference_counter) - set(source_rep)
    add_precision = len(added_good) / len(added) if added else 1.0
    add_recall = len(added_good) / len(added_all) if added_all else 1.0
    add = _f1(add_precision, add_recall)
    return keep, delete_precision, add


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision > 0 or recall > 0 else 0.0


def sentence_sari(source: str, prediction: str, references: Sequence[str]) -> float:
    """SARI in [0, 100] for one example."""
    normalize = lambda text: tokenize_13a(text.lower()).split()  # noqa: E731
    source_tokens = normalize(source)
    prediction_tokens = normalize(prediction)
    reference_tokens = [normalize(r) for r in references]
    scores = []
    for n in range(1, 5):
        scores.append(
            _sari_ngram(
                _ngrams(source_tokens, n),
                _ngrams(prediction_tokens, n),
                [_ngrams(r, n) for r in reference_tokens],
                len(references),
            )
        )
    keep = sum(s[0] for s in scores) / 4
    delete = sum(s[1] for s in scores) / 4
    add = sum(s[2] for s in scores) / 4
    return 100.0 * (keep + delete + add) / 3


# -- chrF -------------------------------------------------------------------


def sentence_chrf(hypothesis: str, reference: str, char_order: int = 6, beta: float = 2.0) -> float:
    """
    Sentence chrF in [0, 100], sacreBLEU-style: whitespace removed, the
    F-beta of each character n-gram order averaged over the orders both
    strings are long enough to have.
    """
    hypothesis, reference = hypothesis.replace(" ", ""), reference.replace(" ", "")
    if not hypothesis and not reference:
        return 100.0
    factor = beta**2
    total, orders = 0.0, 0
    for n in range(1, char_order + 1):
        hyp = Counter(hypothesis[i:i + n] for i in range(len(hypothesis) - n + 1))
        ref = Counter(reference[i:i + n] for i in range(len(reference) - n + 1))
        hyp_total, ref_total = sum(hyp.values()), sum(ref.values())
        if hyp_total == 0 or ref_total == 0:
            continue
        match = sum((hyp & ref).values())
        precision, recall = match / hyp_total, match / ref_total
        denominator = factor * precision + recall
        total += (1 + factor) * precision * recall / denominator if denominator > 0 else 0.0
        orders += 1
    return 100.0 * total / orders if orders else 0.0


# -- preservation -------------------------------------------------------------

_TRAILING = ".,;:!?)]}'\""
_SPAN_PATTERNS = (
    re.compile(r"https?://\S+"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"(?<![\w@])@\w+"),
    re.compile(r"`[^`\n]+`"),
    re.compile(r"(?<![\w.])\d+(?:[.,:/]\d+)*%?"),
)


def protected_spans(text: str) -> list[str]:
    """Spans a rewrite must keep verbatim, deduplicated, in order of appearance."""
    found: list[str] = []
    covered: list[tuple[int, int]] = []
    for pattern in _SPAN_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(s <= start < e for s, e in covered):
                continue  # inside a URL / e-mail / code span already taken
            span = match.group(0).rstrip(_TRAILING) if pattern is _SPAN_PATTERNS[0] else match.group(0)
            if span and span not in found:
                found.append(span)
            covered.append((start, end))
    return found


def span_preservation(source: str, output: str) -> float | None:
    """Fraction of the source's protected spans present in the output; None if it has none."""
    spans = protected_spans(source)
    if not spans:
        return None
    return sum(span in output for span in spans) / len(spans)


# -- degeneration and novelty -------------------------------------------------


def has_loop(text: str, n: int = 3, min_count: int = 3) -> bool:
    tokens = words(text)
    grams = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return any(count >= min_count for count in grams.values())


def novel_word_rate(source: str, output: str) -> float | None:
    output_words = words(output)
    if not output_words:
        return None
    source_words = set(words(source))
    return sum(w not in source_words for w in output_words) / len(output_words)


def distinct_n(texts: Iterable[str], n: int = 2) -> float:
    """Unique word n-grams over total, across a set of outputs (mode-collapse check)."""
    total, unique = 0, set()
    for text in texts:
        tokens = words(text)
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def length_ratio(source: str, output: str) -> float:
    return len(output) / max(len(source), 1)
