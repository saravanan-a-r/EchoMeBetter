/**
 * Turns (adapter, user text) into the encoder's input IDs, following the
 * manifest's prompt frame:
 *
 *   {"token": "..."}      one vocabulary piece by name (frame markers, </s>)
 *   {"styleToken": true}  the style marker of the adapter that will run
 *   {"input": true}       the user's text
 *
 * EchoMeBetter's frame is `<style:X> <text_to_rewrite> text </text_to_rewrite> </s>`,
 * exactly as the adapters were trained (SFT_adapter_rephrase/src/codec.py).
 * Control pieces are ordinary vocabulary entries, so user text that spells
 * one would encode to it and break the frame; such text is refused, as the
 * training data was, instead of handing the model a marker mid-sentence.
 */
import { EchoError } from '../../shared/errors';
import type { StyleId } from '../../shared/styles';
import type { ModelManifest } from '../manifest';
import type { Tokenizer } from '../tokenizer/tokenizer';
import { applyInputTransforms } from './textTransforms';

export class PromptBuilder {
  private readonly ids = new Map<string, number>();
  private readonly reservedIds: ReadonlySet<number>;

  constructor(
    private readonly manifest: ModelManifest,
    private readonly tokenizer: Tokenizer,
  ) {
    const named = [
      ...manifest.prompt.flatMap((segment) => ('token' in segment ? [segment.token] : [])),
      ...Object.values(manifest.adapters).map((adapter) => adapter.styleToken),
      ...manifest.reservedInputTokens,
    ];
    for (const token of named) {
      const id = tokenizer.tokenToId(token);
      if (id === undefined) throw new Error(`model.json: token ${JSON.stringify(token)} is not in the tokenizer's vocabulary`);
      this.ids.set(token, id);
    }
    this.reservedIds = new Set(manifest.reservedInputTokens.map((token) => this.ids.get(token)!));
  }

  /** The encoder input for `text`, rewritten with `adapter` (which must be in the manifest). */
  build(adapter: StyleId, text: string): number[] {
    const styleToken = this.manifest.adapters[adapter]?.styleToken;
    if (styleToken === undefined) throw new Error(`model.json has no ${adapter} adapter`);
    const ids: number[] = [];
    for (const segment of this.manifest.prompt) {
      if ('token' in segment) {
        ids.push(this.ids.get(segment.token)!);
      } else if ('styleToken' in segment) {
        ids.push(this.ids.get(styleToken)!);
      } else {
        const inputIds = this.tokenizer.encode(applyInputTransforms(text, this.manifest.inputTransforms));
        if (inputIds.some((id) => this.reservedIds.has(id))) throw new EchoError('RESERVED_MARKUP');
        ids.push(...inputIds);
      }
    }
    const limit = this.manifest.limits.maxInputTokens;
    if (ids.length > limit) throw new EchoError('INPUT_TOO_LONG', { actual: ids.length, limit });
    return ids;
  }
}
