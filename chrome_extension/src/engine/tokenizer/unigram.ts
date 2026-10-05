/**
 * SentencePiece Unigram segmentation: the highest-scoring way to cover a
 * word with vocabulary pieces (Viterbi over the lattice of matches).
 *
 * A port of HuggingFace `tokenizers`' `Unigram::encode_optimized` and
 * `Unigram::tokenize`, including the parts that decide *which* of several
 * equal-scoring segmentations wins (strict `>` while scanning left to right,
 * prefixes visited shortest first), the unknown-character penalty, fusing
 * runs of unknown characters, and byte fallback. IDs must agree with the
 * Python reference exactly, so tie-breaking is part of the contract.
 */
import type { UnigramModelJson } from './types';

const UNK_PENALTY = 10.0;

interface TrieNode {
  children: Map<string, TrieNode>;
  /** Set when the path from the root to this node spells a vocabulary piece. */
  piece?: string;
}

interface BestPath {
  /** Start offset of the last piece on the best path ending here; -1 = unreachable. */
  startsAt: number;
  id: number;
  score: number;
}

export class UnigramModel {
  private readonly pieces: readonly (readonly [string, number])[];
  private readonly pieceToId = new Map<string, number>();
  private readonly root: TrieNode = { children: new Map() };
  readonly unkId: number | null;
  private readonly unkScore: number;
  private readonly byteFallback: boolean;

  constructor(json: UnigramModelJson) {
    this.pieces = json.vocab;
    this.unkId = json.unk_id;
    this.byteFallback = json.byte_fallback ?? false;
    let minScore = Number.POSITIVE_INFINITY;
    json.vocab.forEach(([piece, score], id) => {
      this.pieceToId.set(piece, id);
      this.insert(piece);
      if (score < minScore) minScore = score;
    });
    this.unkScore = minScore - UNK_PENALTY;
  }

  get vocabSize(): number {
    return this.pieces.length;
  }

  tokenToId(piece: string): number | undefined {
    return this.pieceToId.get(piece);
  }

  idToToken(id: number): string | undefined {
    return this.pieces[id]?.[0];
  }

  private insert(piece: string): void {
    let node = this.root;
    for (const unit of piece) {
      let child = node.children.get(unit);
      if (!child) {
        child = { children: new Map() };
        node.children.set(unit, child);
      }
      node = child;
    }
    node.piece = piece;
  }

  /** Every vocabulary piece that starts at `position`, shortest first. */
  private *prefixesAt(text: string, position: number): Generator<string> {
    let node: TrieNode | undefined = this.root;
    let index = position;
    while (index < text.length) {
      const codePoint = String.fromCodePoint(text.codePointAt(index)!);
      node = node.children.get(codePoint);
      if (!node) return;
      index += codePoint.length;
      if (node.piece !== undefined) yield node.piece;
    }
  }

  /** The best segmentation of one word, as piece strings (unknown runs fused). */
  segment(text: string): string[] {
    if (text.length === 0) return [];
    const best: BestPath[] = Array.from({ length: text.length + 1 }, () => ({ startsAt: -1, id: 0, score: 0 }));
    best[0] = { startsAt: 0, id: 0, score: 0 };

    let startsAt = 0;
    while (startsAt < text.length) {
      const charLength = String.fromCodePoint(text.codePointAt(startsAt)!).length;
      // Like the reference, every code point boundary is processed: with an
      // <unk> token every boundary is reachable, so there is nothing to skip.
      const here = best[startsAt]!;
      {
        let hasSingleCharPiece = false;
        for (const piece of this.prefixesAt(text, startsAt)) {
          const end = startsAt + piece.length;
          const id = this.pieceToId.get(piece)!;
          const candidate = this.pieces[id]![1] + here.score;
          const target = best[end]!;
          if (target.startsAt === -1 || candidate > target.score) {
            best[end] = { startsAt, id, score: candidate };
          }
          if (piece.length === charLength) hasSingleCharPiece = true;
        }
        if (!hasSingleCharPiece && this.unkId !== null) {
          const end = startsAt + charLength;
          const candidate = this.unkScore + here.score;
          const target = best[end]!;
          if (target.startsAt === -1 || candidate > target.score) {
            best[end] = { startsAt, id: this.unkId, score: candidate };
          }
        }
      }
      startsAt += charLength;
    }

    const results: string[] = [];
    let unknownRun: string[] = [];
    let end = text.length;
    while (end > 0) {
      const node = best[end]!;
      if (node.startsAt === -1) throw new Error('unigram lattice has no path (no <unk> token to fall back on)');
      const piece = text.slice(node.startsAt, end);
      if (this.unkId !== null && node.id === this.unkId) {
        unknownRun.push(piece);
      } else {
        if (unknownRun.length > 0) {
          results.push(unknownRun.reverse().join(''));
          unknownRun = [];
        }
        results.push(piece);
      }
      end = node.startsAt;
    }
    if (unknownRun.length > 0) results.push(unknownRun.reverse().join(''));
    return results.reverse();
  }

  /** One word → token IDs. */
  tokenize(text: string): number[] {
    const ids: number[] = [];
    for (const piece of this.segment(text)) {
      const id = this.pieceToId.get(piece);
      if (id !== undefined) {
        ids.push(id);
        continue;
      }
      if (this.byteFallback) {
        const byteIds = byteFallbackIds(piece, this.pieceToId);
        if (byteIds) {
          ids.push(...byteIds);
          continue;
        }
      }
      if (this.unkId === null) throw new Error(`no <unk> token for ${JSON.stringify(piece)}`);
      ids.push(this.unkId);
    }
    return ids;
  }
}

const utf8 = new TextEncoder();

function byteFallbackIds(piece: string, pieceToId: ReadonlyMap<string, number>): number[] | null {
  const ids: number[] = [];
  for (const byte of utf8.encode(piece)) {
    const id = pieceToId.get(`<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`);
    if (id === undefined) return null;
    ids.push(id);
  }
  return ids;
}
