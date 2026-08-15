#!/usr/bin/env bash
#
# Trains the tokenizer at 5 different corpus sizes, back to back, evaluating
# every build on the same-sized eval sample so they can be compared head to
# head. Run this, then hand the full console output back for analysis.
#
# Usage: ./sweep_corpus_sizes.sh [corpus_dir]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CORPUS="${1:-../training_corpus/output}"
CONFIG="config/laptop_16gb.yaml"
SEED=42
EVAL_CORPUS=3000000

# size_label:input_sentence_size:output_dir
RUNS=(
  "2.5M:2500000:output_2_5M"
  "2M:2000000:output_2M"
  "1.5M:1500000:output_1_5M"
  "1.2M:1200000:output_1_2M"
  "1M:1000000:output_1M"
)

echo "=========================================================================="
echo "CORPUS SIZE SWEEP"
echo "corpus:       $CORPUS"
echo "config:       $CONFIG"
echo "seed:         $SEED  (fixed across every run -- same split/shuffle basis)"
echo "eval-corpus:  $EVAL_CORPUS"
echo "=========================================================================="
echo

overall_started=$(date +%s)
declare -a EXIT_CODES=()

for entry in "${RUNS[@]}"; do
  IFS=":" read -r label size out_dir <<< "$entry"

  echo "=========================================================================="
  echo ">>> [$label] input_sentence_size=$size -> output-dir=$out_dir"
  echo "=========================================================================="
  started=$(date +%s)

  python3 train_tokenization.py \
    --corpus "$CORPUS" \
    --config "$CONFIG" \
    --output-dir "$out_dir" \
    --input-sentence-size "$size" \
    --seed "$SEED" \
    --eval-corpus "$EVAL_CORPUS"
  rc=$?
  EXIT_CODES+=("$label:$rc")

  elapsed=$(( $(date +%s) - started ))
  echo
  echo ">>> [$label] finished in ${elapsed}s, exit code $rc"
  echo
done

overall_elapsed=$(( $(date +%s) - overall_started ))

echo "=========================================================================="
echo "SWEEP COMPLETE in ${overall_elapsed}s"
echo "=========================================================================="
for entry in "${EXIT_CODES[@]}"; do
  IFS=":" read -r label rc <<< "$entry"
  if [ "$rc" -eq 0 ]; then
    status="OK"
  else
    status="EXIT $rc (0=ok, 1=hard gate failed, 2=bad config/corpus, 3=training failed, 4=token map failed)"
  fi
  echo "  $label: $status"
done
