/**
 * Build the engine from the downloaded model.
 *
 * Runs inside the inference worker. The base model is read from the
 * ModelStore (the extension's private file system) with byte-level progress
 * and turned into sessions one file at a time, so only one file's bytes are
 * held alongside a session. It then stays in memory until the idle unloader
 * frees it.
 *
 * A style's adapter is loaded only for the rewrite that needs it and released
 * as soon as that rewrite ends: adapters are small and quick to load, and
 * this keeps memory at the base model's size between rewrites. Which adapter
 * a style runs with is looked up from the installed model at each rewrite,
 * so adapters added or removed while the base is loaded take effect at once.
 */
import type * as Ort from 'onnxruntime-web';
import { baseFiles, bytesOf, catalogOf, type AdapterManifest } from '../engine/manifest';
import type { AdapterPair, TensorFactory } from '../engine/onnx/t5Runner';
import { RewriteEngine } from '../engine/rewriteEngine';
import { EchoError } from '../shared/errors';
import type { WorkerConfig } from '../shared/messages';
import { adapterForStyle } from '../shared/modelInstall';
import { styleLabel } from '../shared/styles';
import type { LoadedEngine } from './engineHost';
import type { ModelStore, StoredModel } from './modelStore';

export type OrtModule = Pick<typeof Ort, 'InferenceSession' | 'LoraAdapter' | 'Tensor' | 'env'>;

/** Where the engine finds the installed model, as it is at the moment of asking. */
export interface InstalledModelSource {
  require(): Promise<{ model: StoredModel; store: ModelStore }>;
}

export interface LoadedAdapter extends AdapterPair {
  release(): Promise<void>;
}

export async function loadAdapter(ort: Pick<OrtModule, 'LoraAdapter'>, store: ModelStore, adapter: AdapterManifest): Promise<LoadedAdapter> {
  const encoder = await ort.LoraAdapter.create(await store.read(adapter.files.encoder));
  try {
    const decoder = await ort.LoraAdapter.create(await store.read(adapter.files.decoder));
    return { encoder, decoder, release: async () => void (await Promise.all([encoder.release(), decoder.release()])) };
  } catch (error) {
    await encoder.release();
    throw error;
  }
}

export async function loadEngine(
  ort: OrtModule,
  config: Pick<WorkerConfig, 'wasmBaseUrl' | 'threads'>,
  source: InstalledModelSource,
  onProgress: (fraction: number) => void,
): Promise<LoadedEngine> {
  ort.env.wasm.wasmPaths = config.wasmBaseUrl.endsWith('/') ? config.wasmBaseUrl : `${config.wasmBaseUrl}/`;
  ort.env.wasm.numThreads = config.threads;
  ort.env.wasm.proxy = false; // already off the page's thread: this *is* the worker, and adapters need it off

  const { model, store } = await source.require();
  const { manifest } = model;
  const total = bytesOf(baseFiles(manifest));
  let loaded = 0;
  const tick = (count: number) => {
    loaded += count;
    onProgress(Math.min(1, loaded / total));
  };

  const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(await store.read(manifest.files.tokenizer, tick)));
  const options: Ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    // The adapter weights are graph inputs with empty defaults by design; onnxruntime warns about each one.
    logSeverityLevel: 3,
  };
  const encoder = await ort.InferenceSession.create(await store.read(manifest.files.encoder, tick), options);
  const decoder = await ort.InferenceSession.create(await store.read(manifest.files.decoder, tick), options);
  const engine = new RewriteEngine({
    geometry: manifest.architecture,
    tokenizerJson,
    encoder,
    decoder,
    TensorCtor: ort.Tensor as unknown as TensorFactory,
  });

  return {
    manifest,
    async rewrite(style, text, rewriteOptions) {
      const current = await source.require();
      const adapterId = adapterForStyle(catalogOf(current.model.manifest), style);
      if (!adapterId) throw new EchoError('STYLE_UNAVAILABLE', { style: styleLabel(style) });
      if (!current.model.adapters.includes(adapterId)) {
        throw new EchoError('STYLE_NOT_DOWNLOADED', { style: styleLabel(style), adapter: styleLabel(adapterId) });
      }
      const adapter = await loadAdapter(ort, current.store, current.model.manifest.adapters[adapterId]!);
      try {
        return await engine.rewrite({ manifest: current.model.manifest, adapterId, adapter, text }, rewriteOptions);
      } finally {
        await adapter.release();
      }
    },
    release: () => engine.release(),
  };
}
