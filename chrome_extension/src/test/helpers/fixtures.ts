import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { catalogOf, parseModelManifest } from '../../engine/manifest';
import type { InstalledModelRecord } from '../../shared/modelInstall';
import type { StyleId } from '../../shared/styles';

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
  readonly decoded: string;
}

/** One greedy generation recorded with native onnxruntime, the style's adapter active. */
export interface GenerationCase {
  readonly style: string;
  readonly input: string;
  readonly ids: number[];
  readonly maxNewTokens: number;
  /** EOS included when it was produced. */
  readonly outputIds: number[];
  readonly outputText: string;
  /** The same prompt without any adapter (recorded for the tiny fixtures only). */
  readonly baseOutputIds?: number[];
}

export interface Parity {
  readonly modelId: string;
  readonly tokenizer?: { text: string; ids: number[] }[];
  readonly prompts?: { style: string; input: string; ids: number[] }[];
  readonly generation: GenerationCase[];
}

/** What the tiny fixture's model.json offers: a professional and a grammar adapter, professional as the fallback. */
export const TINY_CATALOG = catalogOf(parseModelManifest(readJsonFixture('tiny-echo/model.json')));

export function installedRecord(sourceUrl: string, adapters: StyleId[] = ['professional']): InstalledModelRecord {
  return { sourceUrl, catalog: TINY_CATALOG, adapters };
}
