import { describe, expect, test } from '@jest/globals';
import * as ort from 'onnxruntime-web';
import { downloadModel, sha256Hex } from '../../../worker/modelDownloader';
import { loadEngine, type OrtModule } from '../../../worker/modelLoader';
import { ModelStore } from '../../../worker/modelStore';
import { readJsonFixture, type Parity } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, modelServer, tinyModelFiles } from '../../helpers/modelServer';

// Real sessions and tensors; the environment the loader configures is a stand-in, so
// the browser-only settings it writes do not leak into onnxruntime under Node.
function ortWithRecordedEnv() {
  const env = { wasm: {} as Record<string, unknown> };
  return { module: { InferenceSession: ort.InferenceSession, Tensor: ort.Tensor, env } as unknown as OrtModule, env };
}

async function downloaded() {
  const store = await ModelStore.open(new MemoryDirectory());
  const server = modelServer(tinyModelFiles('tiny-t5'));
  const model = await downloadModel(
    store,
    FIXTURE_URL,
    { fetch: server.fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 },
    { signal: new AbortController().signal, onProgress: () => undefined },
  );
  return { store, model };
}

describe('loadEngine', () => {
  test('a downloaded model loads from storage and generates exactly what the reference did', async () => {
    const { store, model } = await downloaded();
    const { module, env } = ortWithRecordedEnv();
    const fractions: number[] = [];
    const engine = await loadEngine(module, { wasmBaseUrl: 'chrome-extension://id/ort', threads: 2 }, model, store, (fraction) =>
      fractions.push(fraction),
    );

    expect(env.wasm).toEqual({ wasmPaths: 'chrome-extension://id/ort/', numThreads: 2, proxy: false });
    expect(fractions.at(-1)).toBe(1);
    expect(fractions).toEqual([...fractions].sort((a, b) => a - b));

    const [testCase] = readJsonFixture<Parity>('tiny-t5/parity.json').generation;
    const result = await engine.generateIds(testCase!.ids);
    const expected = testCase!.outputIds.at(-1) === engine.manifest.tokens.eosId ? testCase!.outputIds.slice(0, -1) : testCase!.outputIds;
    expect(result.ids).toEqual(expected);
    await engine.release();
  });

  test('a file that went missing after the download fails the load as "not downloaded"', async () => {
    const { store, model } = await downloaded();
    await store.remove(model.manifest.files.encoder);
    const { module } = ortWithRecordedEnv();
    await expect(loadEngine(module, { wasmBaseUrl: 'x/', threads: 1 }, model, store, () => undefined)).rejects.toMatchObject({
      code: 'MODEL_NOT_DOWNLOADED',
    });
  });
});
