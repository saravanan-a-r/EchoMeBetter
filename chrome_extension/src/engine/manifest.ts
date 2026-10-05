/**
 * `model.json`: everything model-specific, as data.
 *
 * The extension's code knows how to run *a* T5-family encoder-decoder that
 * honours the "t5-cross-kv/1" I/O contract (input/output tensor names and
 * shapes, see onnx/t5Runner.ts). Which checkpoint, how each style is
 * prompted, its special tokens and length limits all come from this file,
 * which ships next to the ONNX files. Replacing the stand-in Flan-T5 with
 * EchoMeBetter is a new model folder with a new manifest; no code changes.
 *
 * The manifest is validated in full before anything is loaded, so a
 * mismatched or hand-edited model fails at load time with a precise message
 * instead of producing wrong text later.
 */
import { STYLE_IDS, type StyleId } from '../shared/styles';

export const SUPPORTED_SCHEMA_VERSION = 1;
export const SUPPORTED_IO_CONTRACT = 't5-cross-kv/1';
export const TEXT_TRANSFORMS = ['escape-spiece-markers'] as const;

export type TextTransformName = (typeof TEXT_TRANSFORMS)[number];

export type PromptSegment =
  | { readonly text: string }
  | { readonly token: string }
  | { readonly input: true };

export interface ModelFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}

export interface ModelManifest {
  readonly schemaVersion: number;
  readonly ioContract: string;
  readonly id: string;
  readonly displayName: string;
  readonly description: string;
  readonly placeholder: boolean;
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
  readonly decode: { readonly cleanUpTokenizationSpaces: boolean };
  readonly inputTransforms: readonly TextTransformName[];
  readonly reservedInputTokens: readonly string[];
  readonly styles: Readonly<Record<StyleId, { readonly prompt: readonly PromptSegment[] }>>;
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
  if (path.includes('..') || path.startsWith('/') || path.includes('://')) {
    throw new ManifestError(`${where}.path must be a plain file name inside the model folder`);
  }
  // The hash is also the file's name in the extension's model storage.
  const sha256 = string(json.sha256, `${where}.sha256`);
  if (!/^[0-9a-f]{64}$/.test(sha256)) throw new ManifestError(`${where}.sha256 must be 64 lowercase hex characters`);
  return { path, bytes: integer(json.bytes, `${where}.bytes`, 1), sha256 };
}

function prompt(value: unknown, where: string): PromptSegment[] {
  if (!Array.isArray(value) || value.length === 0) throw new ManifestError(`${where} must be a non-empty array`);
  const segments = value.map((raw, index): PromptSegment => {
    const segment = object(raw, `${where}[${index}]`);
    if (segment.input === true) return { input: true };
    if (typeof segment.token === 'string') return { token: string(segment.token, `${where}[${index}].token`) };
    if (typeof segment.text === 'string') return { text: string(segment.text, `${where}[${index}].text`) };
    throw new ManifestError(`${where}[${index}] must be {"text"}, {"token"} or {"input": true}`);
  });
  if (segments.filter((segment) => 'input' in segment).length !== 1) {
    throw new ManifestError(`${where} must contain exactly one {"input": true} segment`);
  }
  return segments;
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
  const decode = object(json.decode ?? {}, 'decode');
  const styles = object(json.styles, 'styles');

  const inputTransforms = (json.inputTransforms ?? []) as unknown;
  if (!Array.isArray(inputTransforms) || !inputTransforms.every((name) => (TEXT_TRANSFORMS as readonly unknown[]).includes(name))) {
    throw new ManifestError(`inputTransforms must be a list drawn from ${TEXT_TRANSFORMS.join(', ')}`);
  }
  const reserved = (json.reservedInputTokens ?? []) as unknown;
  if (!Array.isArray(reserved) || !reserved.every((token) => typeof token === 'string')) {
    throw new ManifestError('reservedInputTokens must be a list of strings');
  }

  const parsedStyles = {} as Record<StyleId, { prompt: PromptSegment[] }>;
  for (const id of STYLE_IDS) {
    if (!(id in styles)) throw new ManifestError(`styles.${id} is missing: the model must support every style the menu offers`);
    parsedStyles[id] = { prompt: prompt(object(styles[id], `styles.${id}`).prompt, `styles.${id}.prompt`) };
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
    description: typeof json.description === 'string' ? json.description : '',
    placeholder: json.placeholder === true,
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
    decode: { cleanUpTokenizationSpaces: decode.cleanUpTokenizationSpaces === true },
    inputTransforms: inputTransforms as TextTransformName[],
    reservedInputTokens: reserved as string[],
    styles: parsedStyles,
  };
}

/** Every file the model needs, smallest first. */
export function modelFiles(manifest: ModelManifest): ModelFile[] {
  const { encoder, decoder, tokenizer } = manifest.files;
  return [tokenizer, encoder, decoder];
}

export function totalModelBytes(manifest: ModelManifest): number {
  return modelFiles(manifest).reduce((sum, file) => sum + file.bytes, 0);
}
