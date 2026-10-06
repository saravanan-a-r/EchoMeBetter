/**
 * `model.json`: everything model-specific, as data.
 *
 * EchoMeBetter is a base model plus one LoRA adapter per writing style. The
 * extension's code knows how to run a T5-family encoder-decoder that honours
 * the "t5-cross-kv/1" I/O contract (see onnx/t5Runner.ts) with an adapter
 * active. Which files make up the base and each style's adapter, the prompt
 * frame, the special tokens and the length limits all come from this file,
 * served next to the model files.
 *
 * Adapters are keyed by style. A style without an adapter of its own (still
 * in training) runs with `fallbackAdapter`, under that adapter's style token;
 * a newly trained adapter is one more entry under `adapters`, no code changes.
 *
 * The manifest is validated in full before anything is downloaded or loaded,
 * so a mismatched or hand-edited model fails with a precise message instead
 * of producing wrong text later.
 */
import type { AdapterSummary, ModelCatalog } from '../shared/modelInstall';
import type { ModelSummary } from '../shared/status';
import { isStyleId, type StyleId } from '../shared/styles';

export const SUPPORTED_SCHEMA_VERSION = 2;
export const SUPPORTED_IO_CONTRACT = 't5-cross-kv/1';
export const TEXT_TRANSFORMS = ['escape-spiece-markers'] as const;

export type TextTransformName = (typeof TEXT_TRANSFORMS)[number];

/**
 * The encoder input, one segment at a time: a vocabulary piece by name (frame
 * markers, `</s>`), the active adapter's style token, or the user's text.
 */
export type PromptSegment = { readonly token: string } | { readonly styleToken: true } | { readonly input: true };

export interface ModelFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface AdapterManifest {
  /** The style marker the adapter was trained with. */
  readonly styleToken: string;
  /** One half per graph: each session rejects weights it does not have. */
  readonly files: { readonly encoder: ModelFile; readonly decoder: ModelFile };
}

export interface ModelManifest {
  readonly schemaVersion: number;
  readonly ioContract: string;
  readonly id: string;
  readonly displayName: string;
  readonly precision: string;
  readonly files: { readonly encoder: ModelFile; readonly decoder: ModelFile; readonly tokenizer: ModelFile };
  readonly architecture: {
    readonly numDecoderLayers: number;
    readonly numHeads: number;
    readonly headDim: number;
    readonly vocabSize: number;
  };
  readonly tokens: { readonly decoderStartId: number; readonly eosId: number; readonly padId: number };
  readonly limits: { readonly maxInputTokens: number; readonly maxNewTokens: number };
  readonly generation: { readonly repetitionPenalty: number; readonly noRepeatNgramSize: number };
  readonly inputTransforms: readonly TextTransformName[];
  readonly reservedInputTokens: readonly string[];
  readonly prompt: readonly PromptSegment[];
  readonly adapters: Readonly<Partial<Record<StyleId, AdapterManifest>>>;
  /** The adapter a style without one of its own runs with; null once every style has its own. */
  readonly fallbackAdapter: StyleId | null;
}

export class ManifestError extends Error {
  constructor(message: string) {
    super(`model.json: ${message}`);
    this.name = 'ManifestError';
  }
}

type Json = Record<string, unknown>;

function object(value: unknown, where: string): Json {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) throw new ManifestError(`${where} must be an object`);
  return value as Json;
}

function string(value: unknown, where: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new ManifestError(`${where} must be a non-empty string`);
  return value;
}

function integer(value: unknown, where: string, min: number): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < min) {
    throw new ManifestError(`${where} must be an integer >= ${min}`);
  }
  return value;
}

function file(value: unknown, where: string): ModelFile {
  const json = object(value, where);
  const path = string(json.path, `${where}.path`);
  if (path.split('/').includes('..') || path.startsWith('/') || path.includes('://')) {
    throw new ManifestError(`${where}.path must be a relative path inside the model folder`);
  }
  // The hash is also the file's name in the extension's model storage.
  const sha256 = string(json.sha256, `${where}.sha256`);
  if (!/^[0-9a-f]{64}$/.test(sha256)) throw new ManifestError(`${where}.sha256 must be 64 lowercase hex characters`);
  return { path, bytes: integer(json.bytes, `${where}.bytes`, 1), sha256 };
}

function prompt(value: unknown): PromptSegment[] {
  if (!Array.isArray(value) || value.length === 0) throw new ManifestError('prompt must be a non-empty array');
  const segments = value.map((raw, index): PromptSegment => {
    const segment = object(raw, `prompt[${index}]`);
    if (segment.input === true) return { input: true };
    if (segment.styleToken === true) return { styleToken: true };
    if (typeof segment.token === 'string') return { token: string(segment.token, `prompt[${index}].token`) };
    throw new ManifestError(`prompt[${index}] must be {"token"}, {"styleToken": true} or {"input": true}`);
  });
  for (const kind of ['input', 'styleToken'] as const) {
    if (segments.filter((segment) => kind in segment).length !== 1) {
      throw new ManifestError(`prompt must contain exactly one {"${kind}": true} segment`);
    }
  }
  return segments;
}

function adapters(value: unknown): Partial<Record<StyleId, AdapterManifest>> {
  const json = object(value, 'adapters');
  const parsed: Partial<Record<StyleId, AdapterManifest>> = {};
  for (const [id, raw] of Object.entries(json)) {
    if (!isStyleId(id)) throw new ManifestError(`adapters.${id}: adapters are named after the style they are trained for`);
    const adapter = object(raw, `adapters.${id}`);
    const files = object(adapter.files, `adapters.${id}.files`);
    parsed[id] = {
      styleToken: string(adapter.styleToken, `adapters.${id}.styleToken`),
      files: { encoder: file(files.encoder, `adapters.${id}.files.encoder`), decoder: file(files.decoder, `adapters.${id}.files.decoder`) },
    };
  }
  if (Object.keys(parsed).length === 0) throw new ManifestError('adapters must name at least one adapter');
  return parsed;
}

export function parseModelManifest(raw: unknown): ModelManifest {
  const json = object(raw, 'manifest');
  if (json.schemaVersion !== SUPPORTED_SCHEMA_VERSION) {
    throw new ManifestError(`schemaVersion ${String(json.schemaVersion)} is not supported (expected ${SUPPORTED_SCHEMA_VERSION})`);
  }
  if (json.ioContract !== SUPPORTED_IO_CONTRACT) {
    throw new ManifestError(`ioContract ${String(json.ioContract)} is not supported (expected ${SUPPORTED_IO_CONTRACT})`);
  }

  const files = object(json.files, 'files');
  const architecture = object(json.architecture, 'architecture');
  const tokens = object(json.tokens, 'tokens');
  const limits = object(json.limits, 'limits');
  const generation = object(json.generation ?? {}, 'generation');

  const inputTransforms = (json.inputTransforms ?? []) as unknown;
  if (!Array.isArray(inputTransforms) || !inputTransforms.every((name) => (TEXT_TRANSFORMS as readonly unknown[]).includes(name))) {
    throw new ManifestError(`inputTransforms must be a list drawn from ${TEXT_TRANSFORMS.join(', ')}`);
  }
  const reserved = (json.reservedInputTokens ?? []) as unknown;
  if (!Array.isArray(reserved) || !reserved.every((token) => typeof token === 'string')) {
    throw new ManifestError('reservedInputTokens must be a list of strings');
  }

  const parsedAdapters = adapters(json.adapters);
  const fallback = json.fallbackAdapter ?? null;
  if (fallback !== null && !(isStyleId(fallback) && fallback in parsedAdapters)) {
    throw new ManifestError(`fallbackAdapter ${String(fallback)} is not in adapters`);
  }

  const repetitionPenalty = generation.repetitionPenalty ?? 1;
  if (typeof repetitionPenalty !== 'number' || !(repetitionPenalty > 0)) {
    throw new ManifestError('generation.repetitionPenalty must be a positive number');
  }

  return {
    schemaVersion: SUPPORTED_SCHEMA_VERSION,
    ioContract: SUPPORTED_IO_CONTRACT,
    id: string(json.id, 'id'),
    displayName: string(json.displayName, 'displayName'),
    precision: typeof json.precision === 'string' ? json.precision : 'unknown',
    files: {
      encoder: file(files.encoder, 'files.encoder'),
      decoder: file(files.decoder, 'files.decoder'),
      tokenizer: file(files.tokenizer, 'files.tokenizer'),
    },
    architecture: {
      numDecoderLayers: integer(architecture.numDecoderLayers, 'architecture.numDecoderLayers', 1),
      numHeads: integer(architecture.numHeads, 'architecture.numHeads', 1),
      headDim: integer(architecture.headDim, 'architecture.headDim', 1),
      vocabSize: integer(architecture.vocabSize, 'architecture.vocabSize', 2),
    },
    tokens: {
      decoderStartId: integer(tokens.decoderStartId, 'tokens.decoderStartId', 0),
      eosId: integer(tokens.eosId, 'tokens.eosId', 0),
      padId: integer(tokens.padId, 'tokens.padId', 0),
    },
    limits: {
      maxInputTokens: integer(limits.maxInputTokens, 'limits.maxInputTokens', 1),
      maxNewTokens: integer(limits.maxNewTokens, 'limits.maxNewTokens', 1),
    },
    generation: {
      repetitionPenalty,
      noRepeatNgramSize: integer(generation.noRepeatNgramSize ?? 0, 'generation.noRepeatNgramSize', 0),
    },
    inputTransforms: inputTransforms as TextTransformName[],
    reservedInputTokens: reserved as string[],
    prompt: prompt(json.prompt),
    adapters: parsedAdapters,
    fallbackAdapter: fallback,
  };
}

/** The base model's files, smallest first. */
export function baseFiles(manifest: ModelManifest): ModelFile[] {
  const { encoder, decoder, tokenizer } = manifest.files;
  return [tokenizer, encoder, decoder];
}

export function adapterFiles(adapter: AdapterManifest): ModelFile[] {
  return [adapter.files.encoder, adapter.files.decoder];
}

export function bytesOf(files: readonly ModelFile[]): number {
  return files.reduce((sum, file) => sum + file.bytes, 0);
}

/**
 * Identifies the base weights. Adapters are trained against one base, so a
 * manifest with a different base cannot add adapters to an installed one.
 */
export function baseFingerprint(manifest: ModelManifest): string {
  return baseFiles(manifest)
    .map((file) => file.sha256)
    .join(':');
}

export function summarize(manifest: ModelManifest): ModelSummary {
  return {
    id: manifest.id,
    displayName: manifest.displayName,
    precision: manifest.precision,
    sizeBytes: bytesOf(baseFiles(manifest)),
  };
}

/** What the manifest offers, as the extension's pages show it. */
export function catalogOf(manifest: ModelManifest): ModelCatalog {
  const adapterList: AdapterSummary[] = [];
  for (const [style, adapter] of Object.entries(manifest.adapters) as [StyleId, AdapterManifest][]) {
    adapterList.push({ style, sizeBytes: bytesOf(adapterFiles(adapter)) });
  }
  return { base: baseFingerprint(manifest), model: summarize(manifest), adapters: adapterList, fallbackAdapter: manifest.fallbackAdapter };
}
