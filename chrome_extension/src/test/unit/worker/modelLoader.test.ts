import { describe, expect, jest, test } from '@jest/globals';
import * as ort from 'onnxruntime-web';
import { ModelLibrary } from '../../../worker/modelLibrary';
import { sha256Hex } from '../../../worker/modelDownloader';
import { loadEngine, type OrtModule } from '../../../worker/modelLoader';
import { ModelStore } from '../../../worker/modelStore';
import type { StyleId } from '../../../shared/styles';
import { readJsonFixture, type Parity } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, modelServer, tinyModelFiles } from '../../helpers/modelServer';

const parity = readJsonFixture<Parity>('tiny-echo/parity.json');
/** A professional case that ends in EOS, so its rewrite is returned rather than refused as too long. */
const finished = parity.generation.find((c) => c.style === 'professional' && c.outputIds.at(-1) === 1)!;

// Real sessions, tensors and adapters (counted); the environment the loader configures is a fake,
// so the browser-only settings it writes do not leak into onnxruntime under Node.
function ortRecording() {
  const env = { wasm: {} as Record<string, unknown> };
  const created: ort.LoraAdapter[] = [];
  const LoraAdapter = {
    create: jest.fn(async (bytes: Uint8Array) => {
      const adapter = await ort.LoraAdapter.create(bytes);
      jest.spyOn(adapter, 'release');
      created.push(adapter);
      return adapter;
    }),
  };
  const module = { InferenceSession: ort.InferenceSession, Tensor: ort.Tensor, LoraAdapter, env } as unknown as OrtModule;
  return { module, env, LoraAdapter, created };
}

async function installed(adapters: StyleId[]) {
  const library = new ModelLibrary(FIXTURE_URL, {
    openStore: (() => {
      const root = new MemoryDirectory();
      return () => ModelStore.open(root);
    })(),
    transport: { fetch: modelServer(tinyModelFiles('tiny-echo')).fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 },
    engine: { unload: async () => undefined },
    emit: () => undefined,
  });
  await library.startDownload(adapters);
  return library;
}

describe('loadEngine', () => {
  test('the base loads from storage with progress; each rewrite loads its adapter and releases it after', async () => {
    const library = await installed(['professional']);
    const { module, env, LoraAdapter, created } = ortRecording();
    const fractions: number[] = [];
    const engine = await loadEngine(module, { wasmBaseUrl: 'chrome-extension://id/ort', threads: 2 }, library, (fraction) => fractions.push(fraction));

    expect(env.wasm).toEqual({ wasmPaths: 'chrome-extension://id/ort/', numThreads: 2, proxy: false });
    expect(fractions.at(-1)).toBe(1);
    expect(fractions).toEqual([...fractions].sort((a, b) => a - b));
    expect(LoraAdapter.create).not.toHaveBeenCalled();

    expect((await engine.rewrite('professional', finished.input)).text).toBe(finished.outputText.trim());
    // A style still in training rewrites with the fallback adapter.
    expect((await engine.rewrite('concise', finished.input)).text).toBe(finished.outputText.trim());
    // An encoder and a decoder half per rewrite, each released once the rewrite is done.
    expect(created).toHaveLength(4);
    for (const adapter of created) expect(adapter.release).toHaveBeenCalledTimes(1);
    await engine.release();
  });

  test('a style whose adapter is not downloaded is refused; one added while the base is loaded is used at once', async () => {
    const library = await installed(['professional']);
    const { module, LoraAdapter } = ortRecording();
    const engine = await loadEngine(module, { wasmBaseUrl: 'x/', threads: 1 }, library, () => undefined);
    await expect(engine.rewrite('grammar', finished.input)).rejects.toMatchObject({
      code: 'STYLE_NOT_DOWNLOADED',
      details: { style: 'Grammar', adapter: 'Grammar' },
    });

    await library.startDownload(['grammar']);
    // The fixture's grammar rewrite of this text runs past maxNewTokens, so it is refused as
    // too long -- after running with the grammar adapter.
    await expect(engine.rewrite('grammar', finished.input)).rejects.toMatchObject({ code: 'OUTPUT_TOO_LONG' });
    const grammarFiles = tinyModelFiles('tiny-echo');
    expect(LoraAdapter.create.mock.calls.map(([bytes]) => bytes)).toEqual([
      grammarFiles.get('adapters/grammar/encoder.onnx_adapter'),
      grammarFiles.get('adapters/grammar/decoder.onnx_adapter'),
    ]);
    await engine.release();
  });

  test('a base file that went missing after the download fails the load as "not downloaded"', async () => {
    const library = await installed([]);
    const { store, model } = await library.require();
    await store.remove(model.manifest.files.encoder);
    const { module } = ortRecording();
    await expect(loadEngine(module, { wasmBaseUrl: 'x/', threads: 1 }, library, () => undefined)).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });
});
