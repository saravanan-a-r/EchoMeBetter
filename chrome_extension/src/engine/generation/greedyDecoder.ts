/**
 * Greedy decoding with a KV cache, independent of any inference runtime.
 *
 * The runner encodes the prompt once, then advances one token at a time,
 * handing back the logits and an opaque cache for the next step. Between
 * steps the loop yields to the event loop (each `step` is awaited), which is
 * what lets a cancel message reach the worker mid-generation.
 */
import { EchoError } from '../../shared/errors';
import { argmax, processLogits, type GenerationSettings } from './logitsProcessors';

export interface StepResult<Cache> {
  /** Scores over the vocabulary for the next token. */
  readonly logits: Float32Array;
  readonly cache: Cache;
}

export interface Seq2SeqRunner<Memory, Cache> {
  encode(inputIds: readonly number[]): Promise<Memory>;
  initialCache(): Cache;
  step(tokenId: number, memory: Memory, cache: Cache): Promise<StepResult<Cache>>;
  /** Release anything held by `memory`/`cache`; called once generation ends, however it ends. */
  release?(memory: Memory, cache: Cache): void;
}

export interface GreedyOptions extends GenerationSettings {
  readonly decoderStartId: number;
  readonly eosId: number;
  readonly maxNewTokens: number;
  readonly signal?: AbortSignal;
  readonly onToken?: (generatedCount: number) => void;
}

export interface GreedyResult {
  /** Generated IDs, EOS excluded. */
  readonly ids: number[];
  /** True when EOS was produced; false when maxNewTokens cut generation off. */
  readonly finished: boolean;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) throw new EchoError('CANCELLED');
}

export async function greedyGenerate<Memory, Cache>(
  runner: Seq2SeqRunner<Memory, Cache>,
  inputIds: readonly number[],
  options: GreedyOptions,
): Promise<GreedyResult> {
  throwIfAborted(options.signal);
  const memory = await runner.encode(inputIds);
  let cache = runner.initialCache();
  const sequence = [options.decoderStartId];
  const generated: number[] = [];
  try {
    for (let step = 0; step < options.maxNewTokens; step++) {
      throwIfAborted(options.signal);
      const result = await runner.step(sequence[sequence.length - 1]!, memory, cache);
      cache = result.cache;
      const next = argmax(processLogits(result.logits, sequence, options));
      sequence.push(next);
      if (next === options.eosId) return { ids: generated, finished: true };
      generated.push(next);
      options.onToken?.(generated.length);
    }
    return { ids: generated, finished: false };
  } finally {
    runner.release?.(memory, cache);
  }
}
