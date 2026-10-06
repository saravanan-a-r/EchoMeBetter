import { describe, expect, jest, test } from '@jest/globals';
import { greedyGenerate, type Seq2SeqRunner } from '../../../engine/generation/greedyDecoder';

const EOS = 1;

/** A runner that emits a scripted token per step and records what it was fed. */
function scriptedRunner(script: number[], vocab = 10) {
  const fed: { token: number; cache: number }[] = [];
  const release = jest.fn();
  const runner: Seq2SeqRunner<string, number> = {
    encode: async () => 'memory',
    initialCache: () => 0,
    step: async (token, memory, cache) => {
      expect(memory).toBe('memory');
      fed.push({ token, cache });
      const logits = new Float32Array(vocab);
      logits[script[cache] ?? EOS] = 1;
      return { logits, cache: cache + 1 };
    },
    release,
  };
  return { runner, fed, release };
}

const options = { decoderStartId: 0, eosId: EOS, maxNewTokens: 8, repetitionPenalty: 1, noRepeatNgramSize: 0 };

describe('greedyGenerate', () => {
  test('feeds each token back with the previous step cache and stops at EOS (excluded)', async () => {
    const { runner, fed, release } = scriptedRunner([4, 5, EOS]);
    const result = await greedyGenerate(runner, [7, 8], options);
    expect(result).toEqual({ ids: [4, 5], finished: true });
    expect(fed).toEqual([
      { token: 0, cache: 0 },
      { token: 4, cache: 1 },
      { token: 5, cache: 2 },
    ]);
    expect(release).toHaveBeenCalledTimes(1);
  });

  test('reports unfinished output when maxNewTokens is reached', async () => {
    const { runner } = scriptedRunner([4, 5, 6, 7]);
    expect(await greedyGenerate(runner, [7], { ...options, maxNewTokens: 3 })).toEqual({ ids: [4, 5, 6], finished: false });
  });

  test('cancellation stops before the next step and still releases resources', async () => {
    const { runner, release, fed } = scriptedRunner([4, 5, 6, 7]);
    const controller = new AbortController();
    const pending = greedyGenerate(runner, [7], {
      ...options,
      signal: controller.signal,
      onToken: (count) => count === 2 && controller.abort(),
    });
    await expect(pending).rejects.toMatchObject({ code: 'CANCELLED' });
    expect(fed).toHaveLength(2);
    expect(release).toHaveBeenCalledTimes(1);
  });

  test('an already-aborted signal never touches the model', async () => {
    const { runner, fed } = scriptedRunner([4]);
    const controller = new AbortController();
    controller.abort();
    await expect(greedyGenerate(runner, [7], { ...options, signal: controller.signal })).rejects.toMatchObject({ code: 'CANCELLED' });
    expect(fed).toHaveLength(0);
  });
});
