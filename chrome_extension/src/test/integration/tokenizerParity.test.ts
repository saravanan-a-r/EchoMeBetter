/**
 * The TypeScript tokenizer against HuggingFace `tokenizers` (Python), ID for
 * ID, on a synthetic tokenizer built the way EchoMeBetter's is (identity
 * normalisation, byte fallback, the markers as user-defined pieces).
 * Expected values were produced by the Python library, so any drift in a
 * port of its algorithms shows up here as a concrete probe.
 */
import { describe, expect, test } from '@jest/globals';
import { Tokenizer } from '../../engine/tokenizer/tokenizer';
import { readJsonFixture, type TokenizerProbe } from '../helpers/fixtures';

const json = readJsonFixture<Record<string, unknown>>('tokenizers/byte-fallback.tokenizer.json');

describe('tokenizer matches HuggingFace tokenizers', () => {
  const tokenizer = Tokenizer.fromJson(json);
  const probes = readJsonFixture<TokenizerProbe[]>('tokenizers/byte-fallback.expected.json');

  test.each(probes.map((probe) => [JSON.stringify(probe.text), probe] as const))('%s', (_label, probe) => {
    expect(tokenizer.encode(probe.text)).toEqual(probe.ids);
    expect(tokenizer.decode(probe.ids)).toBe(probe.decoded);
  });
});

test.each([
  ['a normaliser', { normalizer: { type: 'NFKC' } }, /unsupported tokenizer normalizer: NFKC/],
  ['a prefix space', { pre_tokenizer: { type: 'Metaspace', replacement: '▁', prepend_scheme: 'always', split: false } }, /prepend_scheme always/],
  ['splitting into words', { pre_tokenizer: { type: 'Metaspace', replacement: '▁', prepend_scheme: 'never', split: true } }, /Metaspace with split/],
  ['another pre-tokenizer', { pre_tokenizer: { type: 'ByteLevel' } }, /unsupported tokenizer pre_tokenizer: ByteLevel/],
  ['a post-processor', { post_processor: { type: 'TemplateProcessing' } }, /unsupported tokenizer post_processor/],
  ['added tokens', { added_tokens: [{ id: 0, content: '<pad>' }] }, /unsupported tokenizer added_tokens/],
  ['another model', { model: { type: 'BPE' } }, /unsupported tokenizer model: BPE/],
])('a tokenizer.json with %s is refused, not half-loaded', (_label, change, message) => {
  expect(() => Tokenizer.fromJson({ ...json, ...change })).toThrow(message);
});
