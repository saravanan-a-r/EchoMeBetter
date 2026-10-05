/**
 * Added tokens (`</s>`, `<extra_id_0>`, style markers, ...) are cut out of
 * the text before the model ever sees it, exactly as HuggingFace's
 * `AddedVocabulary` does: leftmost-longest matching, with the
 * `single_word` / `lstrip` / `rstrip` flags, tokens with `normalized: false`
 * matched against the raw input and the rest against normalised text.
 */
import type { AddedTokenJson, Split } from './types';
import type { Normalizer } from './normalizers';

interface Candidate {
  readonly token: AddedTokenJson;
  readonly pattern: string;
}

class TokenMatcher {
  /** First character → candidates starting with it, longest first. */
  private readonly byFirstChar = new Map<string, Candidate[]>();

  constructor(candidates: readonly Candidate[]) {
    for (const candidate of candidates) {
      if (candidate.pattern.length === 0) continue;
      const first = candidate.pattern[0]!;
      const bucket = this.byFirstChar.get(first) ?? [];
      bucket.push(candidate);
      this.byFirstChar.set(first, bucket);
    }
    for (const bucket of this.byFirstChar.values()) bucket.sort((a, b) => b.pattern.length - a.pattern.length);
  }

  get isEmpty(): boolean {
    return this.byFirstChar.size === 0;
  }

  /** Non-overlapping leftmost-longest matches, as [start, end, candidate]. */
  *matches(text: string): Generator<readonly [number, number, Candidate]> {
    let index = 0;
    while (index < text.length) {
      const bucket = this.byFirstChar.get(text[index]!);
      const hit = bucket?.find((candidate) => text.startsWith(candidate.pattern, index));
      if (hit) {
        yield [index, index + hit.pattern.length, hit];
        index += hit.pattern.length;
      } else {
        index += 1;
      }
    }
  }
}

const WORD_CHAR = /[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]/u;
const WHITESPACE = /\s/u;

function lastChar(text: string): string | undefined {
  return text.length === 0 ? undefined : String.fromCodePoint(text.codePointAt(text.length - 1)!);
}

export class AddedVocabulary {
  private readonly tokens = new Map<number, AddedTokenJson>();
  private readonly contentToId = new Map<string, number>();
  private readonly raw: TokenMatcher;
  private readonly normalized: TokenMatcher;

  constructor(added: readonly AddedTokenJson[], normalizer: Normalizer | null) {
    const raw: Candidate[] = [];
    const normalized: Candidate[] = [];
    for (const token of added) {
      this.tokens.set(token.id, token);
      this.contentToId.set(token.content, token.id);
      // Two passes, as in the reference, even without a normaliser: raw
      // tokens claim their matches before normalised ones are looked for.
      if (!(token.normalized ?? !token.special)) {
        raw.push({ token, pattern: token.content });
      } else {
        normalized.push({ token, pattern: normalizer ? normalizer(token.content) : token.content });
      }
    }
    this.raw = new TokenMatcher(raw);
    this.normalized = new TokenMatcher(normalized);
  }

  idOf(content: string): number | undefined {
    return this.contentToId.get(content);
  }

  contentOf(id: number): string | undefined {
    return this.tokens.get(id)?.content;
  }

  /** Ids of every added token flagged special (`</s>`, `<pad>`, sentinels, ...). */
  specialIds(): number[] {
    return [...this.tokens.values()].filter((token) => token.special === true).map((token) => token.id);
  }

  isSpecial(id: number): boolean {
    return this.tokens.get(id)?.special === true;
  }

  /**
   * Cut added tokens out of `input`, normalising everything in between.
   * When `allowSpecial` is false, special tokens are not recognised and stay
   * ordinary text (HuggingFace's `encode_special_tokens = True`).
   */
  extractAndNormalize(input: string, normalizer: Normalizer | null, allowSpecial: boolean): Split[] {
    const out: Split[] = [];
    for (const split of this.splitOn(this.raw, { text: input, originalStart: 0 }, allowSpecial)) {
      if (split.id !== undefined) {
        out.push(split);
        continue;
      }
      const text = normalizer ? normalizer(split.text) : split.text;
      if (text.length === 0) continue;
      out.push(...this.splitOn(this.normalized, { text, originalStart: split.originalStart }, allowSpecial));
    }
    return out;
  }

  private splitOn(matcher: TokenMatcher, split: Split, allowSpecial: boolean): Split[] {
    const { text, originalStart } = split;
    if (matcher.isEmpty) return text.length > 0 ? [split] : [];
    const splits: Split[] = [];
    let cursor = 0;
    for (const [matchStart, matchEnd, { token }] of matcher.matches(text)) {
      let start = matchStart;
      let stop = matchEnd;
      if (!allowSpecial && token.special) continue;
      if (token.single_word) {
        const before = lastChar(text.slice(0, start));
        const after = text.slice(stop).codePointAt(0);
        const startsWord = before === undefined || !WORD_CHAR.test(before);
        const endsWord = after === undefined || !WORD_CHAR.test(String.fromCodePoint(after));
        if (!startsWord || !endsWord) continue;
      }
      if (token.lstrip) {
        let newStart = start;
        while (newStart > 0 && WHITESPACE.test(text[newStart - 1]!)) newStart--;
        start = Math.max(newStart, cursor);
      }
      if (token.rstrip) {
        while (stop < text.length && WHITESPACE.test(text[stop]!)) stop++;
      }
      if (cursor < start) splits.push({ text: text.slice(cursor, start), originalStart: originalStart + cursor });
      splits.push({ text: text.slice(start, stop), originalStart: originalStart + start, id: token.id });
      cursor = stop;
    }
    if (cursor < text.length) splits.push({ text: text.slice(cursor), originalStart: originalStart + cursor });
    return splits;
  }
}
