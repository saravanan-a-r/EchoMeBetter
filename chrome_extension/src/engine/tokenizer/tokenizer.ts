/**
 * EchoMeBetter's tokenizer, built from its `tokenizer.json`.
 *
 * The format is the one tokenizer/training/export_hf.py writes from the
 * SentencePiece model: no normaliser; a Metaspace pre-tokenizer that turns
 * spaces into "▁" without adding a prefix or splitting; a Unigram model with
 * byte fallback; no added tokens and no post-processor. Control pieces (`</s>`,
 * the style and frame markers) are ordinary vocabulary pieces, so text that
 * spells one encodes to it -- the prompt builder refuses such user text.
 *
 * A tokenizer.json in any other format is refused when it is loaded.
 */
import { buildDecoder, type Decoder } from './decoders';
import { UnigramModel } from './unigram';
import { UnsupportedTokenizerError, type MetaspaceJson, type TokenizerJson } from './types';

function metaspaceReplacement(json: TokenizerJson['pre_tokenizer']): string {
  if (json?.type !== 'Metaspace') throw new UnsupportedTokenizerError('pre_tokenizer', String(json?.type ?? null));
  const metaspace = json as MetaspaceJson;
  const scheme = metaspace.prepend_scheme ?? (metaspace.add_prefix_space === false ? 'never' : 'always');
  if (scheme !== 'never') throw new UnsupportedTokenizerError('pre_tokenizer', `Metaspace with prepend_scheme ${scheme}`);
  if (metaspace.split !== false) throw new UnsupportedTokenizerError('pre_tokenizer', 'Metaspace with split');
  return metaspace.replacement;
}

export class Tokenizer {
  private readonly replacement: string;
  private readonly decoder: Decoder | null;
  private readonly model: UnigramModel;

  private constructor(json: TokenizerJson) {
    if (json.model?.type !== 'Unigram') throw new UnsupportedTokenizerError('model', String(json.model?.type));
    if (json.normalizer !== null) throw new UnsupportedTokenizerError('normalizer', json.normalizer.type);
    if (json.post_processor !== null) throw new UnsupportedTokenizerError('post_processor', json.post_processor.type);
    if ((json.added_tokens ?? []).length > 0) throw new UnsupportedTokenizerError('added_tokens', `${json.added_tokens!.length} tokens`);
    this.replacement = metaspaceReplacement(json.pre_tokenizer);
    this.decoder = buildDecoder(json.decoder);
    this.model = new UnigramModel(json.model);
  }

  static fromJson(json: unknown): Tokenizer {
    if (typeof json !== 'object' || json === null) throw new Error('tokenizer.json is not an object');
    return new Tokenizer(json as TokenizerJson);
  }

  /** The id of a vocabulary piece, or undefined. */
  tokenToId(token: string): number | undefined {
    return this.model.tokenToId(token);
  }

  encode(text: string): number[] {
    return this.model.tokenize(text.replaceAll(' ', this.replacement));
  }

  decode(ids: readonly number[]): string {
    const tokens: string[] = [];
    for (const id of ids) {
      const token = this.model.idToToken(id);
      if (token !== undefined) tokens.push(token);
    }
    return this.decoder ? this.decoder(tokens).join('') : tokens.join('');
  }
}
