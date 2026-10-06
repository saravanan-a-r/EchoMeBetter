import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export const FIXTURES = `${join(__dirname, '..', 'fixtures')}/`;

export function readFixture(path: string): Buffer {
  return readFileSync(`${FIXTURES}${path}`);
}

export function readJsonFixture<T = unknown>(path: string): T {
  return JSON.parse(readFixture(path).toString('utf8')) as T;
}

export interface TokenizerProbe {
  readonly text: string;
  readonly ids: number[];
  readonly idsUserText: number[];
  readonly decoded: string;
  readonly decodedRaw: string;
}

export interface GenerationCase {
  readonly style: string;
  readonly input: string;
  readonly ids: number[];
  readonly maxNewTokens: number;
  readonly outputIds: number[];
  readonly outputText: string;
}

export interface Parity {
  readonly modelId: string;
  readonly tokenizer: { text: string; ids: number[] }[];
  readonly generation: GenerationCase[];
}
