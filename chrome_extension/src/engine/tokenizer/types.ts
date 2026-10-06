/**
 * The part of HuggingFace's tokenizer.json schema that EchoMeBetter's
 * tokenizer uses (see tokenizer.ts for the format as a whole).
 *
 * Anything outside it is rejected when the tokenizer is built rather than
 * ignored: a step silently skipped would change token IDs without any error,
 * and the model would quietly degrade.
 */

export type PatternJson = { readonly String: string } | { readonly Regex: string };

export interface MetaspaceJson {
  readonly type: 'Metaspace';
  readonly replacement: string;
  readonly prepend_scheme?: string;
  /** Pre-0.15 serialisation of prepend_scheme: false = 'never'. */
  readonly add_prefix_space?: boolean;
  readonly split?: boolean;
}

export type DecoderJson =
  | { readonly type: 'ByteFallback' }
  | { readonly type: 'Fuse' }
  | { readonly type: 'Replace'; readonly pattern: PatternJson; readonly content: string }
  | { readonly type: 'Sequence'; readonly decoders: readonly DecoderJson[] };

export interface UnigramModelJson {
  readonly type: 'Unigram';
  readonly unk_id: number | null;
  readonly vocab: readonly (readonly [string, number])[];
  readonly byte_fallback?: boolean;
}

export interface TokenizerJson {
  readonly added_tokens?: readonly unknown[];
  readonly normalizer: { readonly type: string } | null;
  readonly pre_tokenizer: MetaspaceJson | { readonly type: string } | null;
  readonly post_processor: { readonly type: string } | null;
  readonly decoder: DecoderJson | null;
  readonly model: UnigramModelJson;
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
