/**
 * The TypeScript tokenizer against HuggingFace `tokenizers` (Python), ID for
 * ID. Two synthetic tokenizers cover both families the extension must
 * serve: a T5-style one (precompiled normalisation map, Metaspace with
 * splitting, added tokens using every flag) and an EchoMeBetter-style one
 * (identity normalisation, byte fallback, user-defined pieces). Expected
 * values were produced by the Python library, so any drift in a port of
 * its algorithms shows up here as a concrete probe.
 */
import { describe, expect, test } from '@jest/globals';
import { Tokenizer } from '../../engine/tokenizer/tokenizer';
import { readJsonFixture, type TokenizerProbe } from '../helpers/fixtures';

describe.each(['t5-style', 'byte-fallback'])('%s tokenizer matches HuggingFace tokenizers', (name) => {
  const tokenizer = Tokenizer.fromJson(readJsonFixture(`tokenizers/${name}.tokenizer.json`));
  const probes = readJsonFixture<TokenizerProbe[]>(`tokenizers/${name}.expected.json`);

  test.each(probes.map((probe) => [JSON.stringify(probe.text), probe] as const))('%s', (_label, probe) => {
    expect(tokenizer.encode(probe.text)).toEqual(probe.ids);
    expect(tokenizer.encode(probe.text, { addSpecialTokens: false, allowSpecialTokens: false })).toEqual(probe.idsUserText);
    expect(tokenizer.decode(probe.ids, { skipSpecialTokens: true })).toBe(probe.decoded);
    expect(tokenizer.decode(probe.ids)).toBe(probe.decodedRaw);
  });
});

test('a tokenizer.json using an unsupported component is refused, not half-loaded', () => {
  const json = readJsonFixture<Record<string, unknown>>('tokenizers/t5-style.tokenizer.json');
  expect(() => Tokenizer.fromJson({ ...json, pre_tokenizer: { type: 'ByteLevel' } })).toThrow(/unsupported tokenizer pre_tokenizer: ByteLevel/);
  expect(() => Tokenizer.fromJson({ ...json, model: { type: 'BPE' } })).toThrow(/unsupported tokenizer model: BPE/);
});
