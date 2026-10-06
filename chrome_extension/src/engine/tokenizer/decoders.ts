/**
 * Token strings → text. Each decoder maps a list of token strings to a new
 * list (HuggingFace's `decode_chain`); the final list is concatenated.
 * EchoMeBetter's chain is Replace("▁" → " "), ByteFallback, Fuse.
 */
import { compilePattern, UnsupportedTokenizerError, type DecoderJson } from './types';

export type Decoder = (tokens: string[]) => string[];

const BYTE_TOKEN = /^<0x([0-9A-Fa-f]{2})>$/;
const strictUtf8 = new TextDecoder('utf-8', { fatal: true });

export function buildDecoder(json: DecoderJson | null): Decoder | null {
  if (json === null) return null;
  switch (json.type) {
    case 'ByteFallback':
      return byteFallback;
    case 'Fuse':
      return (tokens) => [tokens.join('')];
    case 'Replace': {
      const pattern = compilePattern(json.pattern);
      return (tokens) => tokens.map((token) => token.replace(pattern, () => json.content));
    }
    case 'Sequence': {
      const steps = json.decoders.map(buildDecoder).filter((step): step is Decoder => step !== null);
      return (tokens) => steps.reduce((current, step) => step(current), tokens);
    }
    default:
      throw new UnsupportedTokenizerError('decoder', (json as { type: string }).type);
  }
}

/** Runs of `<0xNN>` tokens become the UTF-8 text they encode (or one U+FFFD per byte if invalid). */
function byteFallback(tokens: string[]): string[] {
  const out: string[] = [];
  let pending: number[] = [];
  const flush = () => {
    if (pending.length === 0) return;
    try {
      out.push(strictUtf8.decode(Uint8Array.from(pending)));
    } catch {
      for (let i = 0; i < pending.length; i++) out.push('�');
    }
    pending = [];
  };
  for (const token of tokens) {
    const match = BYTE_TOKEN.exec(token);
    if (match) {
      pending.push(Number.parseInt(match[1]!, 16));
    } else {
      flush();
      out.push(token);
    }
  }
  flush();
  return out;
}
