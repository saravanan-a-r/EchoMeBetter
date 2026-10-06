import { describe, expect, test } from '@jest/globals';
import { AddedVocabulary } from '../../../engine/tokenizer/addedTokens';

const token = (id: number, content: string, flags: Record<string, boolean> = {}) => ({ id, content, normalized: false, special: true, ...flags });

describe('AddedVocabulary', () => {
  test('cuts tokens out leftmost-longest', () => {
    const added = new AddedVocabulary([token(1, '<a>'), token(2, '<a><b>')], null);
    expect(added.extractAndNormalize('x<a><b>y<a>', null, true).map((s) => s.id ?? s.text)).toEqual(['x', 2, 'y', 1]);
  });

  test('lstrip/rstrip swallow surrounding whitespace into the token', () => {
    const added = new AddedVocabulary([token(5, '<mask>', { lstrip: true, rstrip: true })], null);
    expect(added.extractAndNormalize('a  <mask>  b', null, true).map((s) => s.id ?? s.text)).toEqual(['a', 5, 'b']);
  });

  test('single_word tokens only match as whole words', () => {
    const added = new AddedVocabulary([token(7, '[R]', { single_word: true, special: false })], null);
    expect(added.extractAndNormalize('x[R]y', null, true).every((s) => s.id === undefined)).toBe(true);
    expect(added.extractAndNormalize('x [R] y', null, true).map((s) => s.id ?? s.text)).toEqual(['x ', 7, ' y']);
  });

  test('with special tokens disallowed, "</s>" in user text stays text', () => {
    const added = new AddedVocabulary([token(1, '</s>')], null);
    expect(added.extractAndNormalize('end</s>', null, false)).toEqual([{ text: 'end</s>', originalStart: 0 }]);
  });

  test('normalized tokens are matched after normalisation', () => {
    const lower = (text: string) => text.toLowerCase();
    const added = new AddedVocabulary([token(9, '<NORM>', { normalized: true, special: false })], lower);
    expect(added.extractAndNormalize('A <Norm> B', lower, true).map((s) => s.id ?? s.text)).toEqual(['a ', 9, ' b']);
  });
});
