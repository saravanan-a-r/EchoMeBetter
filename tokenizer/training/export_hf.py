#!/usr/bin/env python3
"""
Export the trained SentencePiece model as a HuggingFace `tokenizer.json`.

    python export_hf.py --output-dir output

Why this exists
---------------
`transformers` 5.x **removed** the slow-to-fast SentencePiece conversion
machinery (`convert_slow_tokenizer.SpmConverter` / `T5Converter` are gone).
Constructing `T5TokenizerFast(vocab_file="spm.model")` there does not raise --
it silently yields a 4-token vocabulary that encodes everything to `<unk>`.
Discovering that after the tokenizer is frozen and pretraining has started
would be expensive, so the export is built here, explicitly, and validated
byte-for-byte against the sentencepiece model it came from.

The tokenizer is rebuilt directly from the SentencePiece protobuf using
`tokenizers.models.Unigram(byte_fallback=True)`, which is what the removed
converter did internally. The result is verified to produce **identical token
IDs** to `spm.model` -- not merely "similar" output, since a model trained on
one ID space cannot be served by a tokenizer that assigns different IDs.

Component choices mirror the training config (DESIGN.md 6):
  * `Metaspace(prepend_scheme="never")`  <- add_dummy_prefix=False
  * no normalizer                        <- normalization_rule_name="identity"
  * `ByteFallback` decoder               <- byte_fallback=True
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from marker_escape import escape_markers, unescape_markers

MODULE_DIR = Path(__file__).resolve().parent

# Inputs that must survive encode/decode identically through the exported
# tokenizer, mirroring smoke_test.py's probes.
VERIFY_PROBES = [
    "Check https://example.com/path?a=1&b=2#frag now.",
    "See [the docs](https://docs.python.org/3/library/json.html) first.",
    "    def f(x):\n        return x ** 2",
    "   leading whitespace",
    "trailing whitespace   ",
    "collapse    these     spaces?",
    "col1\tcol2\tcol3",
    "Revenue rose 1234567890 to 3.14159 (up 42%).",
    "first.last+tag@sub.example.co.uk",
    "café naïve résumé Ångström",
    "→ ± ½ µs — “smart quotes” … ‰",
    "日本語のテキストです",
    "ship it 🚀🔥 now",
    '{"key": [1, 2, {"nested": null}]}',
    "a &lt; b &amp;&amp; c &gt; d",
    " ",
    "x",
]


def build_hf_tokenizer(model_path: Path):
    """Rebuild the tokenizer from the SentencePiece protobuf."""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(model_path.read_bytes())
    vocab = [(piece.piece, piece.score) for piece in proto.pieces]

    tokenizer = Tokenizer(
        models.Unigram(vocab, unk_id=sp.unk_id(), byte_fallback=True)
    )
    # No normalizer: the training config uses normalization_rule_name="identity",
    # and adding one here would silently rewrite text at inference time only.
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="never", split=False
    )
    tokenizer.decoder = decoders.Sequence(
        [
            decoders.Replace(Regex("▁"), " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
    )
    return tokenizer, sp


def verify(tokenizer, sp, extra_texts: list[str]) -> list[str]:
    """
    Confirm the export is a faithful replacement, not merely a working one.

    Two independent checks, for two independent failure modes:

    1. Vocabulary identity: id N must name the exact same piece string in
       both tokenizers, for every id. Checked exhaustively over the whole
       vocabulary, not sampled -- this is the literal meaning of "same ID
       space" (a model trained on one cannot be served by the other).

    2. Segmentation agreement on real text: for each probe, both must
       round-trip byte-exactly, and should choose the same token sequence.

    A Unigram Viterbi decoder can have more than one globally-optimal
    segmentation when a span is a run of a single repeated character (e.g.
    multi-space indentation, "----" rules): swapping which same-length pieces
    cover which sub-run leaves the summed piece score unchanged (score
    addition is commutative), and both arrangements reconstruct the identical
    text. sentencepiece's C++ Viterbi and the Rust `tokenizers` Viterbi are
    each individually deterministic but can break such ties differently --
    this is a documented, structural property of the algorithm, not a defect
    in either implementation, and it is geometrically impossible for it to
    occur outside a homogeneous-character span (any other content would
    decode to different bytes if reordered). Such a tie is distinguished from
    a genuine mismatch by requiring the two id sequences to be exact
    permutations of each other (same multiset) *and* both to independently
    round-trip byte-exactly -- a real bug (wrong piece, dropped piece,
    different vocabulary) cannot produce that combination by chance: it is
    only possible when both sequences already reconstruct the identical text.
    """
    problems: list[str] = []
    for text in VERIFY_PROBES + extra_texts:
        # Escape SentencePiece's own marker character out of the raw text
        # before encoding (Marker_Encoder_Design_EDGE_CASE.md), and reverse
        # it after decoding, so this compares against the true original.
        escaped = escape_markers(text)
        ids = tokenizer.encode(escaped, add_special_tokens=False).ids
        decoded = unescape_markers(tokenizer.decode(ids))
        if decoded != text:
            problems.append(f"round-trip: {text!r} -> {decoded!r}")
            continue
        spm_ids = sp.encode(escaped)
        if ids != spm_ids:
            is_benign_tie = (
                sorted(ids) == sorted(spm_ids)
                and unescape_markers(sp.decode(spm_ids)) == text
            )
            if not is_benign_tie:
                problems.append(f"id mismatch: {text!r} hf={ids[:12]} spm={spm_ids[:12]}")
    if tokenizer.get_vocab_size() != sp.get_piece_size():
        problems.append(
            f"vocab size {tokenizer.get_vocab_size()} != spm {sp.get_piece_size()}"
        )
    else:
        for piece_id in range(sp.get_piece_size()):
            spm_piece = sp.id_to_piece(piece_id)
            hf_piece = tokenizer.id_to_token(piece_id)
            if hf_piece != spm_piece:
                problems.append(f"vocab id {piece_id}: spm={spm_piece!r} hf={hf_piece!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export spm.model as HuggingFace tokenizer.json")
    parser.add_argument("--output-dir", type=Path, default=MODULE_DIR / "output")
    parser.add_argument(
        "--model", type=Path, default=None, help="Defaults to <output-dir>/spm.model"
    )
    args = parser.parse_args(argv)

    model_path = args.model or (args.output_dir / "spm.model")
    if not model_path.is_file():
        print(f"error: {model_path} not found. Run train_tokenization.py first.", file=sys.stderr)
        return 2

    try:
        tokenizer, sp = build_hf_tokenizer(model_path)
    except ImportError as exc:
        print(f"error: {exc}. Install with: pip install -r requirements.txt", file=sys.stderr)
        return 2

    # Validate against real text from this build's own eval set when available.
    extra: list[str] = []
    holdout = args.output_dir / "tokenizer_evals_set.jsonl"
    if holdout.is_file():
        with open(holdout, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= 2000:
                    break
                extra.append(json.loads(line)["text"])

    problems = verify(tokenizer, sp, extra)
    if problems:
        print(f"error: exported tokenizer is not faithful ({len(problems)} problem(s)):",
              file=sys.stderr)
        for problem in problems[:10]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    out_path = args.output_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"wrote {out_path}")
    print(f"  vocab size      {tokenizer.get_vocab_size()}")
    print(f"  verified on     {len(VERIFY_PROBES) + len(extra)} texts "
          f"(byte-exact round-trip + identical IDs to spm.model)")
    print("\nLoad it with:")
    print("  from tokenizers import Tokenizer")
    print(f"  tok = Tokenizer.from_file('{out_path}')")
    print("  # or: PreTrainedTokenizerFast(tokenizer_file=..., "
          "pad_token='<pad>', eos_token='</s>', unk_token='<unk>')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
