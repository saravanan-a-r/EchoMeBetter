/**
 * A HuggingFace-compatible Unigram tokenizer built from a `tokenizer.json`.
 *
 * Pipeline (HuggingFace `Tokenizer::encode`):
 *   added tokens cut out → normaliser → pre-tokenizer → Unigram → post-processor
 *
 * Covers what T5-family checkpoints use -- Flan-T5's precompiled NFKC map,
 * and EchoMeBetter's identity-normalised, byte-fallback SentencePiece -- and
 * refuses to load a tokenizer.json that needs anything else.
 */
import { AddedVocabulary } from './addedTokens';
import { buildDecoder, type Decoder } from './decoders';
import { buildNormalizer, type Normalizer } from './normalizers';
import { buildPreTokenizer, type PreTokenizer } from './preTokenizers';
import { UnigramModel } from './unigram';
import { UnsupportedTokenizerError, type PostProcessorJson, type TokenizerJson } from './types';

export interface EncodeOptions {
  /** Apply the post-processor (e.g. append `</s>`). Default true, as in HuggingFace. */
  readonly addSpecialTokens?: boolean;
  /** Recognise special tokens written in the text. Default true; false for untrusted user text. */
  readonly allowSpecialTokens?: boolean;
}

export interface DecodeOptions {
  readonly skipSpecialTokens?: boolean;
}

export class Tokenizer {
  private readonly normalizer: Normalizer | null;
  private readonly preTokenizer: PreTokenizer | null;
  private readonly decoder: Decoder | null;
  private readonly model: UnigramModel;
  private readonly added: AddedVocabulary;
  private readonly postProcessor: PostProcessorJson | null;

  private constructor(json: TokenizerJson) {
    if (json.model?.type !== 'Unigram') throw new UnsupportedTokenizerError('model', String(json.model?.type));
    if (json.post_processor && json.post_processor.type !== 'TemplateProcessing') {
      throw new UnsupportedTokenizerError('post_processor', json.post_processor.type);
    }
    this.normalizer = buildNormalizer(json.normalizer);
    this.preTokenizer = buildPreTokenizer(json.pre_tokenizer);
    this.decoder = buildDecoder(json.decoder);
    this.model = new UnigramModel(json.model);
    this.added = new AddedVocabulary(json.added_tokens ?? [], this.normalizer);
    this.postProcessor = json.post_processor;
  }

  static fromJson(json: unknown): Tokenizer {
    if (typeof json !== 'object' || json === null) throw new Error('tokenizer.json is not an object');
    return new Tokenizer(json as TokenizerJson);
  }

  get vocabSize(): number {
    return this.model.vocabSize;
  }

  /** The id of a vocabulary piece or added token, or undefined. */
  tokenToId(token: string): number | undefined {
    return this.added.idOf(token) ?? this.model.tokenToId(token);
  }

  idToToken(id: number): string | undefined {
    return this.added.contentOf(id) ?? this.model.idToToken(id);
  }

  isSpecial(id: number): boolean {
    return this.added.isSpecial(id);
  }

  /** Special added-token ids, `<unk>` excluded: unknown characters legitimately produce it. */
  controlIds(): number[] {
    return this.added.specialIds().filter((id) => id !== this.model.unkId);
  }

  encode(text: string, options: EncodeOptions = {}): number[] {
    const { addSpecialTokens = true, allowSpecialTokens = true } = options;
    const ids: number[] = [];
    for (const split of this.added.extractAndNormalize(text, this.normalizer, allowSpecialTokens)) {
      if (split.id !== undefined) {
        ids.push(split.id);
        continue;
      }
      const words = this.preTokenizer ? this.preTokenizer(split) : [split];
      for (const word of words) ids.push(...this.model.tokenize(word.text));
    }
    return addSpecialTokens ? this.postProcess(ids) : ids;
  }

  decode(ids: readonly number[], options: DecodeOptions = {}): string {
    const { skipSpecialTokens = false } = options;
    const tokens: string[] = [];
    for (const id of ids) {
      if (skipSpecialTokens && this.added.isSpecial(id)) continue;
      const token = this.idToToken(id);
      if (token !== undefined) tokens.push(token);
    }
    return this.decoder ? this.decoder(tokens).join('') : tokens.join(' ');
  }

  private postProcess(ids: number[]): number[] {
    if (!this.postProcessor) return ids;
    const out: number[] = [];
    for (const piece of this.postProcessor.single) {
      if ('Sequence' in piece) {
        out.push(...ids);
      } else {
        const special = this.postProcessor.special_tokens[piece.SpecialToken.id];
        if (!special) throw new Error(`post-processor names unknown special token ${piece.SpecialToken.id}`);
        out.push(...special.ids);
      }
    }
    return out;
  }
}
