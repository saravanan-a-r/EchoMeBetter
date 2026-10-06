/**
 * The whole rewrite path for the loaded base model: prompt → encoder →
 * greedy decoding → text, with the style's adapter active. Runtime-agnostic:
 * it is handed ready sessions, adapters and a Tensor constructor, so it runs
 * the same under onnxruntime-web in the extension's worker and under Node in
 * the test suite.
 *
 * The base (sessions, tokenizer) stays loaded between rewrites. Everything
 * per style comes with each request: the adapter to run with and the
 * manifest that describes it, which may be newer than the one the base was
 * loaded from (adapters are added to an installed base over time).
 */
import { EchoError } from '../shared/errors';
import type { StyleId } from '../shared/styles';
import { greedyGenerate, type GreedyResult } from './generation/greedyDecoder';
import type { ModelManifest } from './manifest';
import { T5Runner, type AdapterPair, type T5Geometry, type TensorFactory } from './onnx/t5Runner';
import { PromptBuilder } from './prompt/promptBuilder';
import { undoInputTransforms } from './prompt/textTransforms';
import { Tokenizer } from './tokenizer/tokenizer';
import type { InferenceSession } from 'onnxruntime-web';

export interface RewriteOptions {
  readonly signal?: AbortSignal;
  readonly onProgress?: (generatedTokens: number) => void;
}

export interface RewriteResult {
  readonly text: string;
  readonly inputTokens: number;
  readonly outputTokens: number;
}

export interface RewriteRequest {
  /** The installed manifest: prompt frame, limits and decoding settings. */
  readonly manifest: ModelManifest;
  /** The adapter the style runs with (its own, or the fallback). */
  readonly adapterId: StyleId;
  readonly adapter: AdapterPair;
  readonly text: string;
}

export interface EngineParts {
  readonly geometry: T5Geometry;
  readonly tokenizerJson: unknown;
  readonly encoder: InferenceSession;
  readonly decoder: InferenceSession;
  readonly TensorCtor: TensorFactory;
}

export class RewriteEngine {
  private readonly tokenizer: Tokenizer;
  private readonly runner: T5Runner;
  private readonly sessions: readonly InferenceSession[];
  private readonly prompts = new WeakMap<ModelManifest, PromptBuilder>();

  constructor(parts: EngineParts) {
    this.sessions = [parts.encoder, parts.decoder];
    this.tokenizer = Tokenizer.fromJson(parts.tokenizerJson);
    this.runner = new T5Runner(parts.TensorCtor, parts.encoder, parts.decoder, parts.geometry);
  }

  /** Encoder input IDs for a request; exposed so tests can check them against the reference. */
  promptIds(manifest: ModelManifest, adapterId: StyleId, text: string): number[] {
    let prompts = this.prompts.get(manifest);
    if (!prompts) {
      prompts = new PromptBuilder(manifest, this.tokenizer);
      this.prompts.set(manifest, prompts);
    }
    return prompts.build(adapterId, text);
  }

  /** Greedy generation with `adapter` active; null runs the plain base model. */
  generateIds(manifest: ModelManifest, inputIds: readonly number[], adapter: AdapterPair | null, options: RewriteOptions = {}): Promise<GreedyResult> {
    const { tokens, generation, limits } = manifest;
    return greedyGenerate(adapter ? this.runner.withAdapter(adapter) : this.runner, inputIds, {
      ...generation,
      decoderStartId: tokens.decoderStartId,
      eosId: tokens.eosId,
      maxNewTokens: limits.maxNewTokens,
      signal: options.signal,
      onToken: options.onProgress,
    });
  }

  /** Generated IDs → text, as the training codec decodes: padding skipped, input transforms undone. */
  decode(manifest: ModelManifest, ids: readonly number[]): string {
    const text = this.tokenizer.decode(ids.filter((id) => id !== manifest.tokens.padId));
    return undoInputTransforms(text, manifest.inputTransforms);
  }

  /** Free the sessions' memory; the engine cannot be used afterwards. */
  async release(): Promise<void> {
    await Promise.all(this.sessions.map((session) => session.release()));
  }

  async rewrite({ manifest, adapterId, adapter, text }: RewriteRequest, options: RewriteOptions = {}): Promise<RewriteResult> {
    const inputIds = this.promptIds(manifest, adapterId, text);
    const result = await this.generateIds(manifest, inputIds, adapter, options);
    if (!result.finished) throw new EchoError('OUTPUT_TOO_LONG', { limit: manifest.limits.maxNewTokens });
    const output = this.decode(manifest, result.ids).trim();
    if (output.length === 0) throw new EchoError('EMPTY_RESULT');
    return { text: output, inputTokens: inputIds.length, outputTokens: result.ids.length };
  }
}
