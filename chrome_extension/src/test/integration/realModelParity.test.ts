/**
 * Checks the model the extension downloads (the folder at the URL in
 * model.source.json), when a developer is serving it: its tokenizer, every
 * style's prompt and the professional adapter's rewrites must match what the
 * packaging tool recorded with native onnxruntime in parity.json. This is the
 * test to run after publishing a new base or adapter. Skipped when nothing
 * is served there (see helpers/hostedModel.ts).
 */
import { afterAll, beforeAll, describe, expect, test } from '@jest/globals';
import { catalogOf } from '../../engine/manifest';
import { Tokenizer } from '../../engine/tokenizer/tokenizer';
import { adapterForStyle } from '../../shared/modelInstall';
import { isStyleId, type StyleId } from '../../shared/styles';
import type { Parity } from '../helpers/fixtures';
import { HOSTED_MODEL_ENV } from '../helpers/hostedModel';
import { fetchModelFile, modelFromUrl, type TestModel } from '../helpers/ortEngine';

const url = process.env[HOSTED_MODEL_ENV];

(url ? describe : describe.skip)(`hosted model at ${url ?? 'model.source.json (not served)'}`, () => {
  let parity: Parity;
  let model: TestModel;
  const json = async <T,>(path: string) => JSON.parse(new TextDecoder().decode(await fetchModelFile(url!, path))) as T;
  const adapterOf = (style: string): StyleId => {
    if (!isStyleId(style)) throw new Error(`unknown style ${style}`);
    return adapterForStyle(catalogOf(model.manifest), style)!;
  };

  beforeAll(async () => {
    parity = await json<Parity>('parity.json');
    model = await modelFromUrl(url!);
  }, 600_000);

  afterAll(async () => {
    await model?.engine.release();
  });

  test('tokenizer reproduces the recorded IDs', async () => {
    const tokenizer = Tokenizer.fromJson(await json('tokenizer.json'));
    for (const probe of parity.tokenizer!) expect(tokenizer.encode(probe.text)).toEqual(probe.ids);
  });

  test('every style builds the recorded prompt, under the style token of the adapter it runs with', () => {
    for (const prompt of parity.prompts!) expect(model.engine.promptIds(model.manifest, adapterOf(prompt.style), prompt.input)).toEqual(prompt.ids);
  });

  test('the base model with an adapter rewrites in onnxruntime-web exactly as native onnxruntime did', async () => {
    for (const testCase of parity.generation) {
      const adapterId = adapterOf(testCase.style);
      const adapter = await model.adapter(adapterId);
      try {
        const result = await model.engine.rewrite({ manifest: model.manifest, adapterId, adapter, text: testCase.input });
        expect(result.text).toBe(testCase.outputText.trim());
      } finally {
        await adapter.release();
      }
    }
  }, 600_000);
});
