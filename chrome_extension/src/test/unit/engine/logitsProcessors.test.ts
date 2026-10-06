import { describe, expect, test } from '@jest/globals';
import { applyRepetitionPenalty, argmax, bannedNgramTokens, processLogits } from '../../../engine/generation/logitsProcessors';

describe('repetition penalty (HuggingFace semantics)', () => {
  test('divides positive and multiplies negative scores of seen tokens', () => {
    const scores = Float32Array.from([2, -2, 2]);
    applyRepetitionPenalty(scores, [0, 1, 1], 2);
    expect(Array.from(scores)).toEqual([1, -4, 2]);
  });
});

describe('no-repeat n-gram', () => {
  test('bans the token that would complete a seen n-gram', () => {
    expect(bannedNgramTokens([0, 5, 6, 5], 2)).toEqual([6]);
    expect(bannedNgramTokens([0, 5, 6, 7, 5, 6], 3)).toEqual([7]);
  });

  test('size 1 bans every token already in the sequence', () => {
    expect(bannedNgramTokens([0, 3, 4], 1).sort()).toEqual([0, 3, 4]);
  });

  test('nothing is banned until the sequence is long enough', () => {
    expect(bannedNgramTokens([0], 3)).toEqual([]);
  });

  test('banned tokens become -Infinity', () => {
    const scores = processLogits(Float32Array.from([0, 0, 9]), [0, 2, 0], { repetitionPenalty: 1, noRepeatNgramSize: 2 });
    expect(scores[2]).toBe(Number.NEGATIVE_INFINITY);
  });
});

test('argmax returns the first index on ties, like torch/numpy', () => {
  expect(argmax(Float32Array.from([1, 3, 3, 2]))).toBe(1);
});
