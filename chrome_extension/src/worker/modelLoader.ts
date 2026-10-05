/**
 * Build a RewriteEngine from the downloaded model.
 *
 * Runs inside the inference worker. The files come from the ModelStore
 * (the extension's private file system), read with byte-level progress, and
 * are turned into sessions one at a time so only one file's bytes are held
 * alongside a session.
 */
import type * as Ort from 'onnxruntime-web';
import { totalModelBytes } from '../engine/manifest';
import { RewriteEngine } from '../engine/rewriteEngine';
import type { TensorFactory } from '../engine/onnx/t5Runner';
import type { WorkerConfig } from '../shared/messages';
import type { ModelStore, StoredModel } from './modelStore';

export type OrtModule = Pick<typeof Ort, 'InferenceSession' | 'Tensor' | 'env'>;

export async function loadEngine(
  ort: OrtModule,
  config: Pick<WorkerConfig, 'wasmBaseUrl' | 'threads'>,
  model: StoredModel,
  store: ModelStore,
  onProgress: (fraction: number) => void,
): Promise<RewriteEngine> {
  ort.env.wasm.wasmPaths = config.wasmBaseUrl.endsWith('/') ? config.wasmBaseUrl : `${config.wasmBaseUrl}/`;
  ort.env.wasm.numThreads = config.threads;
  ort.env.wasm.proxy = false; // already off the page's thread: this *is* the worker

  const { manifest } = model;
  const total = totalModelBytes(manifest);
  let loaded = 0;
  const tick = (count: number) => {
    loaded += count;
    onProgress(Math.min(1, loaded / total));
  };

  const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(await store.read(manifest.files.tokenizer, tick)));
  const options: Ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  };
  const encoder = await ort.InferenceSession.create(await store.read(manifest.files.encoder, tick), options);
  const decoder = await ort.InferenceSession.create(await store.read(manifest.files.decoder, tick), options);

  return new RewriteEngine({
    manifest,
    tokenizerJson,
    encoder,
    decoder,
    TensorCtor: ort.Tensor as unknown as TensorFactory,
  });
}
