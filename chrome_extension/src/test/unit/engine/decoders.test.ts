import { describe, expect, test } from '@jest/globals';
import { buildDecoder } from '../../../engine/tokenizer/decoders';

describe('decoders', () => {
  test('ByteFallback joins byte tokens into UTF-8 and marks invalid bytes', () => {
    const decode = buildDecoder({ type: 'ByteFallback' })!;
    expect(decode(['a', '<0xC3>', '<0xA9>', 'b'])).toEqual(['a', 'é', 'b']);
    expect(decode(['<0xC3>', 'x'])).toEqual(['�', 'x']);
  });

  test('Metaspace drops every marker in the first token only (reference behaviour)', () => {
    const decode = buildDecoder({ type: 'Metaspace', replacement: '▁', prepend_scheme: 'always' })!;
    expect(decode(['▁Hello', '▁world'])).toEqual(['Hello', ' world']);
  });

  test('Metaspace with prepend_scheme "never" keeps a leading space', () => {
    const decode = buildDecoder({ type: 'Metaspace', replacement: '▁', prepend_scheme: 'never' })!;
    expect(decode(['▁Hi'])).toEqual([' Hi']);
  });

  test('Sequence of Replace, ByteFallback, Fuse (EchoMeBetter decoder)', () => {
    const decode = buildDecoder({
      type: 'Sequence',
      decoders: [{ type: 'Replace', pattern: { Regex: '▁' }, content: ' ' }, { type: 'ByteFallback' }, { type: 'Fuse' }],
    })!;
    expect(decode(['Hi', '▁there', '<0x21>'])).toEqual(['Hi there!']);
  });

  test('unsupported decoders are refused', () => {
    expect(() => buildDecoder({ type: 'WordPiece' } as never)).toThrow(/unsupported tokenizer decoder/);
  });
});
