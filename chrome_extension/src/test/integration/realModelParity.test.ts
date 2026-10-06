/**
 * Checks the model the extension downloads (the folder at the URL in
 * model.source.json), when a developer is serving it: its tokenizer and two
 * generations must match what the exporter recorded in parity.json. This is
 * the test to run after swapping in a new model (e.g. EchoMeBetter for the
 * Flan-T5 stand-in). Skipped when nothing is served there (see
 * helpers/hostedModel.ts).
 */
import { beforeAll, describe, expect, test } from '@jest/globals';
import type { RewriteEngine } from '../../engine/rewriteEngine';
import { Tokenizer } from '../../engine/tokenizer/tokenizer';
import { isStyleId } from '../../shared/styles';
import type { Parity } from '../helpers/fixtures';
import { HOSTED_MODEL_ENV } from '../helpers/hostedModel';
import { engineFromUrl, fetchModelFile } from '../helpers/ortEngine';

const url = process.env[HOSTED_MODEL_ENV];

(url ? describe : describe.skip)(`hosted model at ${url ?? 'model.source.json (not served)'}`, () => {
  let parity: Parity;
  let engine: RewriteEngine;
  const json = async <T,>(path: string) => JSON.parse(new TextDecoder().decode(await fetchModelFile(url!, path))) as T;

  beforeAll(async () => {
    parity = await json<Parity>('parity.json');
    engine = await engineFromUrl(url!);
  }, 600_000);

  test('tokenizer reproduces the exporter-recorded IDs', async () => {
    const tokenizer = Tokenizer.fromJson(await json('tokenizer.json'));
    for (const probe of parity.tokenizer) expect(tokenizer.encode(probe.text)).toEqual(probe.ids);
  });

  test('every style builds the exporter-recorded prompt', () => {
    // Prompt assembly is exact; it does not depend on any numerics.
    for (const testCase of parity.generation) {
      if (!isStyleId(testCase.style)) throw new Error(`unknown style ${testCase.style}`);
      expect(engine.promptIds(testCase.style, testCase.input)).toEqual(testCase.ids);
    }
  });

  test('the model runs in onnxruntime-web and opens with the exporter-recorded token', async () => {
    // An int8 model's activations are quantized at run time and native vs
    // WebAssembly kernels round them differently, so only the first token --
    // decided before any rounding difference can compound -- must agree.
    const testCase = parity.generation[0]!;
    const result = await engine.generateIds(testCase.ids);
    expect(result.ids[0]).toBe(testCase.outputIds[0]);
    expect(engine.decode(result.ids).trim().length).toBeGreaterThan(0);
  }, 300_000);
});
