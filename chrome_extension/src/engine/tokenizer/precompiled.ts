/**
 * SentencePiece's precompiled normalisation map (`Precompiled` in a
 * HuggingFace tokenizer.json), e.g. T5's NFKC-plus-cleanup rules.
 *
 * The blob is base64 of:
 *   u32 LE   trie size in bytes
 *   u32 LE[] a darts-clone double-array trie over UTF-8 byte sequences
 *   bytes    the replacement strings, NUL-separated; a trie value is a byte
 *            offset into this region
 *
 * The lookup and the grapheme-by-grapheme walk mirror the `spm_precompiled`
 * crate that HuggingFace `tokenizers` uses, quirks included: the shortest
 * matching prefix wins, and short graphemes are looked up whole before
 * falling back to one code point at a time. Token IDs must match the
 * reference byte for byte, so this is a port, not a reinterpretation.
 */

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true });

function hasLeaf(unit: number): boolean {
  return ((unit >>> 8) & 1) === 1;
}

function value(unit: number): number {
  return unit & 0x7fffffff;
}

function label(unit: number): number {
  return (unit & 0x800000ff) >>> 0;
}

function offset(unit: number): number {
  return (unit >>> 10) << ((unit & (1 << 9)) >>> 6);
}

export class PrecompiledCharsMap {
  private readonly trie: Uint32Array;
  private readonly normalized: Uint8Array;
  private readonly segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });

  constructor(base64: string) {
    const bytes = base64ToBytes(base64);
    if (bytes.length < 4) throw new Error('precompiled charsmap is truncated');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const trieSize = view.getUint32(0, true);
    if (trieSize % 4 !== 0 || 4 + trieSize > bytes.length) throw new Error('precompiled charsmap has an invalid trie size');
    this.trie = new Uint32Array(trieSize / 4);
    for (let i = 0; i < this.trie.length; i++) this.trie[i] = view.getUint32(4 + i * 4, true);
    this.normalized = bytes.subarray(4 + trieSize);
  }

  /** Values of every trie key that is a prefix of `key`, shortest first. */
  private commonPrefixSearch(key: Uint8Array): number[] {
    const results: number[] = [];
    let nodePos = 0;
    let unit = this.trie[nodePos]!;
    nodePos ^= offset(unit);
    for (let i = 0; i < key.length; i++) {
      const byte = key[i]!;
      nodePos ^= byte;
      unit = this.trie[nodePos] ?? 0;
      if (label(unit) !== byte) return results;
      nodePos ^= offset(unit);
      if (hasLeaf(unit)) results.push(value(this.trie[nodePos] ?? 0));
    }
    return results;
  }

  /** The replacement for `chunk`, or undefined when the map leaves it alone. */
  transform(chunk: string): string | undefined {
    const results = this.commonPrefixSearch(encoder.encode(chunk));
    if (results.length === 0) return undefined;
    const start = results[0]!;
    let end = start;
    while (end < this.normalized.length && this.normalized[end] !== 0) end++;
    return decoder.decode(this.normalized.subarray(start, end));
  }

  normalize(text: string): string {
    let out = '';
    for (const { segment: grapheme } of this.segmenter.segment(text)) {
      if (utf8Length(grapheme) < 6) {
        const replacement = this.transform(grapheme);
        if (replacement !== undefined) {
          out += replacement;
          continue;
        }
      }
      for (const codePoint of grapheme) {
        out += this.transform(codePoint) ?? codePoint;
      }
    }
    return out;
  }
}

function utf8Length(text: string): number {
  let length = 0;
  for (const codePoint of text) {
    const cp = codePoint.codePointAt(0)!;
    length += cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4;
  }
  return length;
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
