/**
 * HuggingFace's greedy-search logits processors, for a single hypothesis,
 * applied in HuggingFace's order (repetition penalty, then no-repeat n-gram).
 *
 * `sequence` is the decoder input so far, decoder-start token included --
 * the same `input_ids` HuggingFace passes its processors -- so a manifest's
 * generation settings mean exactly what they mean in `model.generate`.
 */

export interface GenerationSettings {
  readonly repetitionPenalty: number;
  readonly noRepeatNgramSize: number;
}

export function applyRepetitionPenalty(scores: Float32Array, sequence: readonly number[], penalty: number): void {
  if (penalty === 1) return;
  for (const token of new Set(sequence)) {
    const score = scores[token]!;
    scores[token] = score < 0 ? score * penalty : score / penalty;
  }
}

export function bannedNgramTokens(sequence: readonly number[], size: number): number[] {
  if (size <= 0 || sequence.length + 1 < size) return [];
  const prefixStart = sequence.length + 1 - size;
  const banned: number[] = [];
  for (let start = 0; start + size <= sequence.length; start++) {
    let matches = true;
    for (let k = 0; k < size - 1; k++) {
      if (sequence[start + k] !== sequence[prefixStart + k]) {
        matches = false;
        break;
      }
    }
    if (matches) banned.push(sequence[start + size - 1]!);
  }
  return banned;
}

export function processLogits(scores: Float32Array, sequence: readonly number[], settings: GenerationSettings): Float32Array {
  applyRepetitionPenalty(scores, sequence, settings.repetitionPenalty);
  for (const token of bannedNgramTokens(sequence, settings.noRepeatNgramSize)) scores[token] = Number.NEGATIVE_INFINITY;
  return scores;
}

/** Index of the largest score; the first one on ties, like numpy/torch argmax. */
export function argmax(scores: Float32Array): number {
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i]! > scores[best]!) best = i;
  return best;
}
