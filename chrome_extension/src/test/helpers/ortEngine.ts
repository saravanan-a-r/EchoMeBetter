/**
 * Build a RewriteEngine from model files, running onnxruntime-web under
 * Node -- the same runtime the extension's worker uses.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { parseModelManifest, type ModelManifest } from '../../engine/manifest';
import type { TensorFactory } from '../../engine/onnx/t5Runner';
import { RewriteEngine } from '../../engine/rewriteEngine';

ort.env.wasm.numThreads = 1;

type ReadFile = (path: string) => Uint8Array | Promise<Uint8Array>;

async function engineFromFiles(manifest: ModelManifest, read: ReadFile): Promise<RewriteEngine> {
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' } as const;
  return new RewriteEngine({
    manifest,
    tokenizerJson: JSON.parse(new TextDecoder().decode(await read(manifest.files.tokenizer.path))),
    encoder: await ort.InferenceSession.create(await read(manifest.files.encoder.path), options),
    decoder: await ort.InferenceSession.create(await read(manifest.files.decoder.path), options),
    TensorCtor: ort.Tensor as unknown as TensorFactory,
  });
}

export function engineFromFolder(folder: string): Promise<RewriteEngine> {
  const manifest = parseModelManifest(JSON.parse(readFileSync(join(folder, 'model.json'), 'utf8')));
  return engineFromFiles(manifest, (path) => readFileSync(join(folder, path)));
}

/** Fetch `path` from the model folder served at `baseUrl`. */
export async function fetchModelFile(baseUrl: string, path: string): Promise<Uint8Array> {
  const response = await fetch(new URL(path, baseUrl));
  if (!response.ok) throw new Error(`${new URL(path, baseUrl).toString()}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

export async function engineFromUrl(baseUrl: string): Promise<RewriteEngine> {
  const manifest = parseModelManifest(JSON.parse(new TextDecoder().decode(await fetchModelFile(baseUrl, 'model.json'))));
  return engineFromFiles(manifest, (path) => fetchModelFile(baseUrl, path));
}
