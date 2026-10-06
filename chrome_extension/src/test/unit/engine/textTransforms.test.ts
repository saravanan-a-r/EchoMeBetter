import { describe, expect, test } from '@jest/globals';
import { applyInputTransforms, escapeSpieceMarkers, undoInputTransforms, unescapeSpieceMarkers } from '../../../engine/prompt/textTransforms';

describe('SentencePiece marker escaping (mirror of tokenizer/training/marker_escape.py)', () => {
  test.each(['plain text', '▃▁▁ block art', ' escape char', '▁', ''])('round-trips %j', (text) => {
    expect(unescapeSpieceMarkers(escapeSpieceMarkers(text))).toBe(text);
  });

  test('a literal marker never reaches the tokenizer', () => {
    expect(escapeSpieceMarkers('a▁b')).not.toContain('▁');
  });

  test('transforms are undone in reverse order', () => {
    const escaped = applyInputTransforms('x▁y', ['escape-spiece-markers']);
    expect(undoInputTransforms(escaped, ['escape-spiece-markers'])).toBe('x▁y');
  });
});
