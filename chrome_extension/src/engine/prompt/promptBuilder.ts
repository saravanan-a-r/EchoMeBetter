/**
 * Turns (style, user text) into the encoder's input IDs, following the
 * style's prompt segments from the manifest:
 *
 *   {"text": "..."}   template words, tokenized on their own
 *   {"token": "..."}  one vocabulary piece by name (style markers, </s>)
 *   {"input": true}   the user's text, tokenized with special tokens disabled
 *
 * Segments are tokenized separately so special-token parsing can be switched
 * off for the user's text. That alone is not enough: a Unigram vocabulary
 * also holds control pieces such as "</s>", and HuggingFace's tokenizer will
 * still match them as ordinary pieces. So user text whose IDs include a
 * control token (or a manifest-reserved marker) is refused instead of
 * silently handing the model an end-of-sequence in the middle of a sentence.
 * The exporter proves, per model, that segment-wise tokenizing equals
 * tokenizing the whole prompt naturally.
 */
import { EchoError } from '../../shared/errors';
import type { StyleId } from '../../shared/styles';
import type { ModelManifest } from '../manifest';
import type { Tokenizer } from '../tokenizer/tokenizer';
import { applyInputTransforms } from './textTransforms';

export class PromptBuilder {
  private readonly resolved: ReadonlyMap<string, number>;
  private readonly reservedIds: ReadonlySet<number>;

  constructor(
    private readonly manifest: ModelManifest,
    private readonly tokenizer: Tokenizer,
  ) {
    const resolved = new Map<string, number>();
    const resolve = (token: string, where: string) => {
      const id = tokenizer.tokenToId(token);
      if (id === undefined) throw new Error(`${where}: token ${JSON.stringify(token)} is not in the tokenizer's vocabulary`);
      resolved.set(token, id);
      return id;
    };
    for (const [style, { prompt }] of Object.entries(manifest.styles)) {
      for (const segment of prompt) if ('token' in segment) resolve(segment.token, `styles.${style}`);
    }
    this.reservedIds = new Set([
      ...tokenizer.controlIds(),
      ...manifest.reservedInputTokens.map((token) => resolve(token, 'reservedInputTokens')),
    ]);
    this.resolved = resolved;
  }

  build(style: StyleId, text: string): number[] {
    const userText = applyInputTransforms(text, this.manifest.inputTransforms);
    const ids: number[] = [];
    for (const segment of this.manifest.styles[style].prompt) {
      if ('text' in segment) {
        ids.push(...this.tokenizer.encode(segment.text, { addSpecialTokens: false }));
      } else if ('token' in segment) {
        ids.push(this.resolved.get(segment.token)!);
      } else {
        const inputIds = this.tokenizer.encode(userText, { addSpecialTokens: false, allowSpecialTokens: false });
        if (inputIds.some((id) => this.reservedIds.has(id))) throw new EchoError('RESERVED_MARKUP');
        ids.push(...inputIds);
      }
    }
    const limit = this.manifest.limits.maxInputTokens;
    if (ids.length > limit) throw new EchoError('INPUT_TOO_LONG', { actual: ids.length, limit });
    return ids;
  }
}
