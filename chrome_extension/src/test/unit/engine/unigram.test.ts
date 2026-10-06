import { describe, expect, test } from '@jest/globals';
import { UnigramModel } from '../../../engine/tokenizer/unigram';

const vocab = (pieces: [string, number][]) => ({ type: 'Unigram' as const, unk_id: 0, vocab: [['<unk>', 0] as [string, number], ...pieces] });

describe('UnigramModel', () => {
  test('picks the highest-scoring segmentation, not the longest piece', () => {
    const model = new UnigramModel(vocab([['ab', -10], ['a', -1], ['b', -1], ['abc', -20], ['c', -1]]));
    expect(model.segment('abc')).toEqual(['a', 'b', 'c']);
  });

  test('on equal scores the first path found wins (left-to-right, shortest prefix first)', () => {
    // "aa"+"a" and "a"+"aa" both score -2; the reference keeps the path reaching each position first.
    const model = new UnigramModel(vocab([['a', -1], ['aa', -1]]));
    expect(model.segment('aaa')).toEqual(['a', 'aa']);
  });

  test('a run of unknown characters becomes one <unk>', () => {
    const model = new UnigramModel(vocab([['a', -1]]));
    expect(model.segment('a☃☃a')).toEqual(['a', '☃☃', 'a']);
    expect(model.tokenize('a☃☃a')).toEqual([1, 0, 1]);
  });

  test('byte fallback spells unknown text as UTF-8 byte pieces', () => {
    const bytes = Array.from({ length: 256 }, (_, b) => [`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`, -5] as [string, number]);
    const model = new UnigramModel({ ...vocab([['a', -1], ...bytes]), byte_fallback: true });
    const ids = model.tokenize('é');
    expect(ids.map((id) => model.idToToken(id))).toEqual(['<0xC3>', '<0xA9>']);
  });

  test('astral characters are handled as whole code points', () => {
    const model = new UnigramModel(vocab([['🚀', -1], ['x', -1]]));
    expect(model.segment('x🚀x')).toEqual(['x', '🚀', 'x']);
  });

  test('empty input has no tokens', () => {
    expect(new UnigramModel(vocab([['a', -1]])).tokenize('')).toEqual([]);
  });
});
