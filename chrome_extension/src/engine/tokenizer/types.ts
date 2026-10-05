/**
 * The subset of HuggingFace's tokenizer.json schema this tokenizer reads.
 *
 * Anything outside the subset is rejected when the tokenizer is built rather
 * than ignored: a normaliser or pre-tokenizer silently skipped would change
 * token IDs without any error, and the model would quietly degrade.
 */

export interface AddedTokenJson {
  readonly id: number;
  readonly content: string;
  readonly single_word?: boolean;
  readonly lstrip?: boolean;
  readonly rstrip?: boolean;
  readonly normalized?: boolean;
  readonly special?: boolean;
}

export type PatternJson = { readonly String: string } | { readonly Regex: string };

export type NormalizerJson =
  | { readonly type: 'Sequence'; readonly normalizers: readonly NormalizerJson[] }
  | { readonly type: 'Precompiled'; readonly precompiled_charsmap: string | null }
  | { readonly type: 'Replace'; readonly pattern: PatternJson; readonly content: string }
  | { readonly type: 'NFC' | 'NFD' | 'NFKC' | 'NFKD' | 'Lowercase' }
  | { readonly type: 'Strip'; readonly strip_left: boolean; readonly strip_right: boolean }
  | { readonly type: 'Prepend'; readonly prepend: string };

export type PrependScheme = 'always' | 'first' | 'never';

export interface MetaspaceJson {
  readonly type: 'Metaspace';
  readonly replacement: string;
  readonly prepend_scheme?: PrependScheme;
  /** Pre-0.15 serialisation of prepend_scheme: true = 'always', false = 'never'. */
  readonly add_prefix_space?: boolean;
  readonly split?: boolean;
}

export type PreTokenizerJson =
  | MetaspaceJson
  | { readonly type: 'WhitespaceSplit' }
  | { readonly type: 'Sequence'; readonly pretokenizers: readonly PreTokenizerJson[] };

export type DecoderJson =
  | MetaspaceJson
  | { readonly type: 'ByteFallback' }
  | { readonly type: 'Fuse' }
  | { readonly type: 'Replace'; readonly pattern: PatternJson; readonly content: string }
  | { readonly type: 'Strip'; readonly content: string; readonly start: number; readonly stop: number }
  | { readonly type: 'Sequence'; readonly decoders: readonly DecoderJson[] };

export type TemplatePieceJson =
  | { readonly Sequence: { readonly id: 'A' | 'B'; readonly type_id: number } }
  | { readonly SpecialToken: { readonly id: string; readonly type_id: number } };

export interface PostProcessorJson {
  readonly type: 'TemplateProcessing';
  readonly single: readonly TemplatePieceJson[];
  readonly special_tokens: Readonly<Record<string, { readonly id: string; readonly ids: readonly number[] }>>;
}

export interface UnigramModelJson {
  readonly type: 'Unigram';
  readonly unk_id: number | null;
  readonly vocab: readonly (readonly [string, number])[];
  readonly byte_fallback?: boolean;
}

export interface TokenizerJson {
  readonly added_tokens?: readonly AddedTokenJson[];
  readonly normalizer: NormalizerJson | null;
  readonly pre_tokenizer: PreTokenizerJson | null;
  readonly post_processor: PostProcessorJson | null;
  readonly decoder: DecoderJson | null;
  readonly model: UnigramModelJson;
}

/** A stretch of input on its way through the pipeline. `id` is set once it is an added token. */
export interface Split {
  readonly text: string;
  /** Offset of this split in the original input, used by Metaspace's 'first' scheme. */
  readonly originalStart: number;
  readonly id?: number;
}

export class UnsupportedTokenizerError extends Error {
  constructor(component: string, type: string) {
    super(`unsupported tokenizer ${component}: ${type}`);
    this.name = 'UnsupportedTokenizerError';
  }
}

export function compilePattern(pattern: PatternJson): RegExp {
  if ('String' in pattern) return new RegExp(escapeRegExp(pattern.String), 'gu');
  return new RegExp(pattern.Regex, 'gu');
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
