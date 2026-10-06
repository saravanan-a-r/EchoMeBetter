import { describe, expect, test } from '@jest/globals';
import { buildDecoder } from '../../../engine/tokenizer/decoders';

describe('decoders', () => {
  test('ByteFallback joins byte tokens into UTF-8 and marks invalid bytes', () => {
    const decode = buildDecoder({ type: 'ByteFallback' })!;
    expect(decode(['a', '<0xC3>', '<0xA9>', 'b'])).toEqual(['a', 'é', 'b']);
    expect(decode(['<0xC3>', 'x'])).toEqual(['�', 'x']);
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
