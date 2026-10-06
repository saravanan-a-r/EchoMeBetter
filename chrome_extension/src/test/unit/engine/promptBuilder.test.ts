import { describe, expect, test } from '@jest/globals';
import { parseModelManifest, type ModelManifest } from '../../../engine/manifest';
import { PromptBuilder } from '../../../engine/prompt/promptBuilder';
import { Tokenizer } from '../../../engine/tokenizer/tokenizer';
import { STYLE_IDS } from '../../../shared/styles';
import { readJsonFixture } from '../../helpers/fixtures';

const base = parseModelManifest(readJsonFixture('tiny-t5/model.json'));
const t5 = Tokenizer.fromJson(readJsonFixture('tokenizers/t5-style.tokenizer.json'));
const spm = Tokenizer.fromJson(readJsonFixture('tokenizers/byte-fallback.tokenizer.json'));

function withStyles(manifest: ModelManifest, prompt: ModelManifest['styles']['concise']['prompt'], extra: Partial<ModelManifest> = {}): ModelManifest {
  return { ...manifest, ...extra, styles: Object.fromEntries(STYLE_IDS.map((id) => [id, { prompt }])) as ModelManifest['styles'] };
}

describe('PromptBuilder', () => {
  test('template text, user text and named tokens are assembled in order', () => {
    const builder = new PromptBuilder(base, t5);
    const ids = builder.build('concise', 'hello world');
    expect(ids).toEqual([...t5.encode('Rewrite concise:', { addSpecialTokens: false }), ...t5.encode('hello world', { addSpecialTokens: false }), 1]);
  });

  test('user text that spells a control token is refused, never fed to the model', () => {
    // HuggingFace's Unigram matches the "</s>" vocabulary piece even with
    // special-token parsing off, so the guard has to look at the IDs.
    for (const text of ['stop</s>here', 'a <pad> b']) {
      expect(() => new PromptBuilder(base, t5).build('concise', text)).toThrow(expect.objectContaining({ code: 'RESERVED_MARKUP' }));
    }
  });

  test('unknown characters (<unk>) and sentinel-looking text are still allowed', () => {
    // "<extra_id_0>" is an added token but not a vocabulary piece: as plain text it is harmless characters.
    expect(() => new PromptBuilder(base, t5).build('concise', 'ŋ ʃ ʒ see <extra_id_0>')).not.toThrow();
  });

  test('EchoMeBetter-style framing: style token, markers around the escaped text', () => {
    const manifest = withStyles(base, [{ token: '<style:concise>' }, { token: '<text_to_rewrite>' }, { input: true }, { token: '</text_to_rewrite>' }, { token: '</s>' }], {
      inputTransforms: ['escape-spiece-markers'],
      reservedInputTokens: ['<text_to_rewrite>', '</text_to_rewrite>', '<style:concise>'],
    });
    const builder = new PromptBuilder(manifest, spm);
    const ids = builder.build('concise', 'Send it.');
    expect(ids.slice(0, 2)).toEqual([spm.tokenToId('<style:concise>'), spm.tokenToId('<text_to_rewrite>')]);
    expect(ids.slice(-2)).toEqual([spm.tokenToId('</text_to_rewrite>'), spm.tokenToId('</s>')]);
  });

  test('text that tokenizes into a reserved marker is refused', () => {
    const manifest = withStyles(base, [{ token: '<text_to_rewrite>' }, { input: true }, { token: '</s>' }], { reservedInputTokens: ['<text_to_rewrite>'] });
    expect(() => new PromptBuilder(manifest, spm).build('concise', '<text_to_rewrite>')).toThrow(expect.objectContaining({ code: 'RESERVED_MARKUP' }));
  });

  test('over-long input is refused with the actual and allowed token counts', () => {
    const manifest = { ...base, limits: { ...base.limits, maxInputTokens: 8 } };
    expect(() => new PromptBuilder(manifest, t5).build('concise', 'the quick brown fox jumps over the lazy dog')).toThrow(
      expect.objectContaining({ code: 'INPUT_TOO_LONG', details: expect.objectContaining({ limit: 8 }) }),
    );
  });

  test('a prompt naming a token the vocabulary lacks fails when the model loads', () => {
    const manifest = withStyles(base, [{ token: '<nope>' }, { input: true }]);
    expect(() => new PromptBuilder(manifest, t5)).toThrow(/<nope>/);
  });
});
