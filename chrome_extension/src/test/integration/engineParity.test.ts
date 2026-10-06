/**
 * The whole rewrite path -- prompt assembly, onnxruntime-web, the KV-cached
 * greedy loop, logits processors, decoding -- against what the Python
 * reference produced for the same exported model: a tiny random T5 exported
 * through the real exporter, with tied embeddings, repetition penalty and
 * no-repeat n-gram switched on. The exporter proved the reference equals
 * HuggingFace `generate`, so agreement here means the extension reproduces
 * HuggingFace generation token for token.
 *
 * Exact agreement is asserted on the float32 export. int8 graphs quantize
 * activations at run time, and onnxruntime's native (x86/ARM) and
 * WebAssembly kernels round those differently -- on this fixture's random
 * weights enough to change even the first token -- so the int8 test checks
 * the quantized operators load and run in onnxruntime-web.
 */
import { beforeAll, describe, expect, test } from '@jest/globals';
import type { RewriteEngine } from '../../engine/rewriteEngine';
import { EchoError } from '../../shared/errors';
import { isStyleId } from '../../shared/styles';
import { FIXTURES, readJsonFixture, type Parity } from '../helpers/fixtures';
import { engineFromFolder } from '../helpers/ortEngine';

const parity = readJsonFixture<Parity>('tiny-t5/parity.json');
const int8Parity = readJsonFixture<Parity>('tiny-t5-int8/parity.json');
let engine: RewriteEngine;
let int8Engine: RewriteEngine;

beforeAll(async () => {
  engine = await engineFromFolder(`${FIXTURES}tiny-t5`);
  int8Engine = await engineFromFolder(`${FIXTURES}tiny-t5-int8`);
});

describe('tiny T5 fixture: TypeScript engine == Python reference', () => {
  test.each(parity.generation.map((c) => [`${c.style}: ${c.input}`, c] as const))('%s', async (_label, testCase) => {
    if (!isStyleId(testCase.style)) throw new Error(`fixture has unknown style ${testCase.style}`);
    expect(engine.promptIds(testCase.style, testCase.input)).toEqual(testCase.ids);
    const result = await engine.generateIds(testCase.ids);
    const expected = testCase.outputIds.at(-1) === engine.manifest.tokens.eosId ? testCase.outputIds.slice(0, -1) : testCase.outputIds;
    expect(result.ids).toEqual(expected);
    expect(engine.decode(result.ids)).toBe(testCase.outputText);
  });
});

test('int8 graphs (MatMulInteger, quantized Gather) run under onnxruntime-web', async () => {
  const vocab = int8Engine.manifest.architecture.vocabSize;
  for (const testCase of int8Parity.generation) {
    if (!isStyleId(testCase.style)) throw new Error(`fixture has unknown style ${testCase.style}`);
    expect(int8Engine.promptIds(testCase.style, testCase.input)).toEqual(testCase.ids);
    const result = await int8Engine.generateIds(testCase.ids);
    expect(result.ids.length).toBeLessThanOrEqual(testCase.maxNewTokens);
    expect(result.ids.every((id) => Number.isInteger(id) && id >= 0 && id < vocab)).toBe(true);
  }
});

test('generation stops at the next step once the signal aborts', async () => {
  const controller = new AbortController();
  const steps: number[] = [];
  const pending = engine.generateIds(parity.generation[0]!.ids, {
    signal: controller.signal,
    onProgress: (count) => {
      steps.push(count);
      if (count === 2) controller.abort();
    },
  });
  await expect(pending).rejects.toMatchObject({ code: 'CANCELLED' });
  expect(steps).toEqual([1, 2]);
});

test('a rewrite cut off by maxNewTokens is refused rather than returned truncated', async () => {
  // The fixture's random weights never emit EOS within 16 tokens.
  const testCase = parity.generation.find((c) => !c.outputIds.includes(engine.manifest.tokens.eosId))!;
  await expect(engine.rewrite('professional', testCase.input)).rejects.toBeInstanceOf(EchoError);
  await expect(engine.rewrite('professional', testCase.input)).rejects.toMatchObject({ code: 'OUTPUT_TOO_LONG' });
});

test('input longer than the model limit is refused before any inference', async () => {
  const long = Array.from({ length: 80 }, () => 'report').join(' ');
  expect(() => engine.promptIds('concise', long)).toThrow(expect.objectContaining({ code: 'INPUT_TOO_LONG' }));
});
