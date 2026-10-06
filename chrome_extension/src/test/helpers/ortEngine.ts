/**
 * The base model and its adapters from model files, running onnxruntime-web
 * under Node -- the same runtime the extension's worker uses.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { parseModelManifest, type ModelManifest } from '../../engine/manifest';
import type { TensorFactory } from '../../engine/onnx/t5Runner';
import { RewriteEngine } from '../../engine/rewriteEngine';
import type { StyleId } from '../../shared/styles';
import type { LoadedAdapter } from '../../worker/modelLoader';

ort.env.wasm.numThreads = 1;

type ReadFile = (path: string) => Uint8Array | Promise<Uint8Array>;

export interface TestModel {
  readonly manifest: ModelManifest;
  readonly engine: RewriteEngine;
  /** `style`'s adapter, loaded; the caller releases it. */
  adapter(style: StyleId): Promise<LoadedAdapter>;
}

async function modelFromFiles(manifest: ModelManifest, read: ReadFile): Promise<TestModel> {
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', logSeverityLevel: 3 } as const;
  const engine = new RewriteEngine({
    geometry: manifest.architecture,
    tokenizerJson: JSON.parse(new TextDecoder().decode(await read(manifest.files.tokenizer.path))),
    encoder: await ort.InferenceSession.create(await read(manifest.files.encoder.path), options),
    decoder: await ort.InferenceSession.create(await read(manifest.files.decoder.path), options),
    TensorCtor: ort.Tensor as unknown as TensorFactory,
  });
  return {
    manifest,
    engine,
    async adapter(style) {
      const files = manifest.adapters[style]!.files;
      const encoder = await ort.LoraAdapter.create(await read(files.encoder.path));
      const decoder = await ort.LoraAdapter.create(await read(files.decoder.path));
      return { encoder, decoder, release: async () => void (await Promise.all([encoder.release(), decoder.release()])) };
    },
  };
}

export function modelFromFolder(folder: string): Promise<TestModel> {
  const manifest = parseModelManifest(JSON.parse(readFileSync(join(folder, 'model.json'), 'utf8')));
  return modelFromFiles(manifest, (path) => new Uint8Array(readFileSync(join(folder, path))));
}

/** Fetch `path` from the model folder served at `baseUrl`. */
export async function fetchModelFile(baseUrl: string, path: string): Promise<Uint8Array> {
  const response = await fetch(new URL(path, baseUrl));
  if (!response.ok) throw new Error(`${new URL(path, baseUrl).toString()}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

export async function modelFromUrl(baseUrl: string): Promise<TestModel> {
  const manifest = parseModelManifest(JSON.parse(new TextDecoder().decode(await fetchModelFile(baseUrl, 'model.json'))));
  return modelFromFiles(manifest, (path) => fetchModelFile(baseUrl, path));
}
