/**
 * The whole rewrite path for one loaded model: prompt → encoder → greedy
 * decoding → text. Runtime-agnostic: it is handed ready sessions and a
 * Tensor constructor, so it runs the same under onnxruntime-web in the
 * extension's worker and under Node in the test suite.
 */
import { EchoError } from '../shared/errors';
import type { StyleId } from '../shared/styles';
import { greedyGenerate } from './generation/greedyDecoder';
import type { ModelManifest } from './manifest';
import { T5Runner, type TensorFactory } from './onnx/t5Runner';
import { PromptBuilder } from './prompt/promptBuilder';
import { cleanUpTokenization, undoInputTransforms } from './prompt/textTransforms';
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

export interface EngineParts {
  readonly manifest: ModelManifest;
  readonly tokenizerJson: unknown;
  readonly encoder: InferenceSession;
  readonly decoder: InferenceSession;
  readonly TensorCtor: TensorFactory;
}

export class RewriteEngine {
  readonly manifest: ModelManifest;
  private readonly tokenizer: Tokenizer;
  private readonly prompts: PromptBuilder;
  private readonly runner: T5Runner;
  private readonly sessions: readonly InferenceSession[];

  constructor(parts: EngineParts) {
    this.manifest = parts.manifest;
    this.sessions = [parts.encoder, parts.decoder];
    this.tokenizer = Tokenizer.fromJson(parts.tokenizerJson);
    this.prompts = new PromptBuilder(parts.manifest, this.tokenizer);
    this.runner = new T5Runner(parts.TensorCtor, parts.encoder, parts.decoder, parts.manifest.architecture);
  }

  /** Encoder input IDs for a request; exposed so tests can check them against the reference. */
  promptIds(style: StyleId, text: string): number[] {
    return this.prompts.build(style, text);
  }

  async generateIds(inputIds: readonly number[], options: RewriteOptions = {}) {
    const { tokens, generation, limits } = this.manifest;
    return greedyGenerate(this.runner, inputIds, {
      ...generation,
      decoderStartId: tokens.decoderStartId,
      eosId: tokens.eosId,
      maxNewTokens: limits.maxNewTokens,
      signal: options.signal,
      onToken: options.onProgress,
    });
  }

  decode(ids: readonly number[]): string {
    let text = this.tokenizer.decode(ids, { skipSpecialTokens: true });
    if (this.manifest.decode.cleanUpTokenizationSpaces) text = cleanUpTokenization(text);
    return undoInputTransforms(text, this.manifest.inputTransforms);
  }

  /** Free the sessions' memory; the engine cannot be used afterwards. */
  async release(): Promise<void> {
    await Promise.all(this.sessions.map((session) => session.release()));
  }

  async rewrite(style: StyleId, text: string, options: RewriteOptions = {}): Promise<RewriteResult> {
    const inputIds = this.promptIds(style, text);
    const result = await this.generateIds(inputIds, options);
    if (!result.finished) throw new EchoError('OUTPUT_TOO_LONG', { limit: this.manifest.limits.maxNewTokens });
    const output = this.decode(result.ids).trim();
    if (output.length === 0) throw new EchoError('EMPTY_RESULT');
    return { text: output, inputTokens: inputIds.length, outputTokens: result.ids.length };
  }
}
