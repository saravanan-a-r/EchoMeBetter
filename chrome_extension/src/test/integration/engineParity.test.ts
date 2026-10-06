/**
 * The whole rewrite path -- prompt assembly, onnxruntime-web with a LoRA
 * adapter active, the KV-cached greedy loop, logits processors, decoding --
 * against native onnxruntime on the same files: a tiny random T5 in
 * EchoMeBetter's package format (LoRA-ready base graphs, a professional and a
 * grammar adapter, the other styles falling back to professional), with
 * repetition penalty and no-repeat n-gram switched on. Without an adapter the
 * fixture's graphs were proven equal to HuggingFace `generate`.
 *
 * Every case also records the plain base model's output, which differs: an
 * adapter that silently failed to apply would show up here.
 */
import { afterAll, beforeAll, describe, expect, test } from '@jest/globals';
import { adapterForStyle } from '../../shared/modelInstall';
import { catalogOf } from '../../engine/manifest';
import { isStyleId, type StyleId } from '../../shared/styles';
import { FIXTURES, readJsonFixture, type GenerationCase, type Parity } from '../helpers/fixtures';
import { modelFromFolder, type TestModel } from '../helpers/ortEngine';

const parity = readJsonFixture<Parity>('tiny-echo/parity.json');
const int8Parity = readJsonFixture<Parity>('tiny-echo-int8/parity.json');
let model: TestModel;
let int8Model: TestModel;

beforeAll(async () => {
  model = await modelFromFolder(`${FIXTURES}tiny-echo`);
  int8Model = await modelFromFolder(`${FIXTURES}tiny-echo-int8`);
});

afterAll(async () => {
  await model.engine.release();
  await int8Model.engine.release();
});

function adapterOf(testModel: TestModel, testCase: GenerationCase): StyleId {
  if (!isStyleId(testCase.style)) throw new Error(`fixture has unknown style ${testCase.style}`);
  return adapterForStyle(catalogOf(testModel.manifest), testCase.style)!;
}

/** Generated IDs as the engine reports them: EOS is not part of the text. */
function withoutEos(testModel: TestModel, ids: readonly number[]): number[] {
  return ids.at(-1) === testModel.manifest.tokens.eosId ? ids.slice(0, -1) : [...ids];
}

async function generate(testModel: TestModel, testCase: GenerationCase, withAdapter: boolean) {
  const adapter = withAdapter ? await testModel.adapter(adapterOf(testModel, testCase)) : null;
  try {
    return await testModel.engine.generateIds(testModel.manifest, testCase.ids, adapter);
  } finally {
    await adapter?.release();
  }
}

describe('tiny EchoMeBetter fixture: onnxruntime-web == native onnxruntime', () => {
  test.each(parity.generation.map((c) => [`${c.style}: ${c.input}`, c] as const))('%s', async (_label, testCase) => {
    expect(model.engine.promptIds(model.manifest, adapterOf(model, testCase), testCase.input)).toEqual(testCase.ids);
    expect((await generate(model, testCase, true)).ids).toEqual(withoutEos(model, testCase.outputIds));
    expect((await generate(model, testCase, false)).ids).toEqual(withoutEos(model, testCase.baseOutputIds!));
  });

  test('a style without an adapter of its own runs with the fallback, under its style token', () => {
    const concise = parity.generation.find((c) => c.style === 'concise')!;
    const professional = parity.generation.find((c) => c.style === 'professional' && c.input === concise.input)!;
    expect(adapterOf(model, concise)).toBe('professional');
    expect(concise.outputIds).toEqual(professional.outputIds);
  });
});

test('int8 base graphs (int8 weights behind DequantizeLinear) run with adapters under onnxruntime-web', async () => {
  // Native and WebAssembly onnxruntime take different kernels for int8 graphs, and on this
  // fixture's random weights that changes tokens within the first few steps -- so here the
  // graphs must load, run with an adapter, and the adapter must change the output.
  const vocab = int8Model.manifest.architecture.vocabSize;
  for (const testCase of int8Parity.generation) {
    expect(int8Model.engine.promptIds(int8Model.manifest, adapterOf(int8Model, testCase), testCase.input)).toEqual(testCase.ids);
    const adapted = await generate(int8Model, testCase, true);
    expect(adapted.ids.length).toBeLessThanOrEqual(testCase.maxNewTokens);
    expect(adapted.ids.every((id) => Number.isInteger(id) && id >= 0 && id < vocab)).toBe(true);
    expect(adapted.ids).not.toEqual((await generate(int8Model, testCase, false)).ids);
  }
});

test('an adapter released after one rewrite can be loaded again, and the base keeps working', async () => {
  const testCase = parity.generation[0]!;
  const first = await generate(model, testCase, true);
  const again = await generate(model, testCase, true);
  expect(again.ids).toEqual(first.ids);
  expect((await generate(model, testCase, false)).ids).toEqual(withoutEos(model, testCase.baseOutputIds!));
});

test('rewrite returns the decoded text of a finished generation', async () => {
  const testCase = parity.generation.find((c) => c.outputIds.at(-1) === model.manifest.tokens.eosId)!;
  const adapterId = adapterOf(model, testCase);
  const adapter = await model.adapter(adapterId);
  try {
    const result = await model.engine.rewrite({ manifest: model.manifest, adapterId, adapter, text: testCase.input });
    expect(result.text).toBe(testCase.outputText.trim());
  } finally {
    await adapter.release();
  }
});

test('generation stops at the next step once the signal aborts', async () => {
  const controller = new AbortController();
  const steps: number[] = [];
  const pending = model.engine.generateIds(model.manifest, parity.generation[1]!.ids, null, {
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
  // The fixture's random weights do not emit EOS within 16 tokens for this case.
  const testCase = parity.generation.find((c) => c.outputIds.at(-1) !== model.manifest.tokens.eosId)!;
  const adapterId = adapterOf(model, testCase);
  const adapter = await model.adapter(adapterId);
  try {
    await expect(model.engine.rewrite({ manifest: model.manifest, adapterId, adapter, text: testCase.input })).rejects.toMatchObject({
      code: 'OUTPUT_TOO_LONG',
    });
  } finally {
    await adapter.release();
  }
});

test('input longer than the model limit is refused before any inference', () => {
  const long = Array.from({ length: 80 }, () => 'report').join(' ');
  expect(() => model.engine.promptIds(model.manifest, 'professional', long)).toThrow(expect.objectContaining({ code: 'INPUT_TOO_LONG' }));
});
