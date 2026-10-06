import { describe, expect, test } from '@jest/globals';
import { parseModelManifest } from '../../../engine/manifest';
import { PromptBuilder } from '../../../engine/prompt/promptBuilder';
import { Tokenizer } from '../../../engine/tokenizer/tokenizer';
import { readJsonFixture } from '../../helpers/fixtures';

const manifest = parseModelManifest(readJsonFixture('tiny-echo/model.json'));
const tokenizer = Tokenizer.fromJson(readJsonFixture('tiny-echo/tokenizer.json'));
const id = (token: string) => tokenizer.tokenToId(token)!;

describe('PromptBuilder', () => {
  test('the frame the adapters were trained on: style token, markers around the text, end of sequence', () => {
    const ids = new PromptBuilder(manifest, tokenizer).build('grammar', 'hello world');
    expect(ids).toEqual([id('<style:grammar>'), id('<text_to_rewrite>'), ...tokenizer.encode('hello world'), id('</text_to_rewrite>'), id('</s>')]);
  });

  test('the style token is the one of the adapter that runs', () => {
    const ids = new PromptBuilder(manifest, tokenizer).build('professional', 'hello');
    expect(ids[0]).toBe(id('<style:professional>'));
  });

  test('user text that spells a control piece is refused, never fed to the model', () => {
    // Control pieces are ordinary vocabulary entries, so this text would encode to them.
    for (const text of ['stop</s>here', 'a <pad> b', 'see <text_to_rewrite> this', '<style:concise>']) {
      expect(() => new PromptBuilder(manifest, tokenizer).build('professional', text)).toThrow(expect.objectContaining({ code: 'RESERVED_MARKUP' }));
    }
  });

  test('characters outside the vocabulary are still allowed', () => {
    expect(() => new PromptBuilder(manifest, tokenizer).build('professional', 'ŋ ʃ ʒ 日本語 🚀')).not.toThrow();
  });

  test('a literal "▁" in the text is escaped before tokenizing', () => {
    const ids = new PromptBuilder(manifest, tokenizer).build('professional', 'a▁b');
    expect(ids.slice(2, -2)).toEqual(tokenizer.encode('ab'));
  });

  test('over-long input is refused with the actual and allowed token counts', () => {
    const short = { ...manifest, limits: { ...manifest.limits, maxInputTokens: 8 } };
    expect(() => new PromptBuilder(short, tokenizer).build('professional', 'the quick brown fox jumps over the lazy dog')).toThrow(
      expect.objectContaining({ code: 'INPUT_TOO_LONG', details: expect.objectContaining({ limit: 8 }) }),
    );
  });

  test('a manifest naming a token the vocabulary lacks fails when the model loads', () => {
    expect(() => new PromptBuilder({ ...manifest, prompt: [{ styleToken: true }, { token: '<nope>' }, { input: true }] }, tokenizer)).toThrow(/<nope>/);
  });
});
