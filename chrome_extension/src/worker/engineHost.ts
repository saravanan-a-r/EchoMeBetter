/**
 * The worker's state: the base model, loaded on first need, and a strictly
 * serial queue of rewrite jobs (each loads and releases its style's adapter).
 *
 * Serial on purpose: the model already uses several threads for one job, and
 * two concurrent jobs would only slow each other down while doubling memory.
 * A job can be cancelled while queued (it never starts) or while running
 * (the decoding loop stops at its next step).
 *
 * An engine whose GPU failed mid-rewrite can't run again: it is dropped, and
 * the job runs once more on a freshly loaded one.
 */
import type { RunningOn } from '../shared/compute';
import { EchoError, toErrorPayload } from '../shared/errors';
import type { EngineEvent } from '../shared/messages';
import type { StyleId } from '../shared/styles';
import { summarize, type ModelManifest } from '../engine/manifest';
import type { RewriteOptions, RewriteResult } from '../engine/rewriteEngine';

export interface LoadedEngine {
  /** The manifest the base was loaded from. */
  readonly manifest: ModelManifest;
  readonly runningOn: RunningOn;
  rewrite(style: StyleId, text: string, options?: RewriteOptions): Promise<RewriteResult>;
  release(): Promise<void>;
}

/** Thrown by an engine that must not run again; the host loads a new one and runs the job once more. */
export class EngineBrokenError extends Error {
  constructor(readonly reason: unknown) {
    super(reason instanceof Error ? reason.message : String(reason));
    this.name = 'EngineBrokenError';
  }
}

export type EngineLoader = (onProgress: (fraction: number) => void) => Promise<LoadedEngine>;

export class EngineHost {
  private engine: LoadedEngine | null = null;
  private loading: Promise<LoadedEngine> | null = null;
  private queue: Promise<void> = Promise.resolve();
  private readonly controllers = new Map<string, AbortController>();
  /** Jobs waiting on the model load; each hears the load's progress. */
  private readonly waitingForModel = new Set<string>();

  constructor(
    private readonly load: EngineLoader,
    private readonly emit: (event: EngineEvent) => void,
  ) {}

  /** Load the model once; concurrent callers share the same attempt, and a failed attempt can be retried. */
  ensureEngine(): Promise<LoadedEngine> {
    if (this.engine) return Promise.resolve(this.engine);
    if (!this.loading) {
      this.emit({ type: 'status', status: { state: 'loading', progress: 0 } });
      let lastReported = 0;
      this.loading = this.load((fraction) => {
        // Throttle to whole percents: hundreds of chunks would flood three message hops.
        if (fraction - lastReported < 0.01 && fraction < 1) return;
        lastReported = fraction;
        this.emit({ type: 'status', status: { state: 'loading', progress: fraction } });
        for (const jobId of this.waitingForModel) {
          this.emit({ type: 'job-phase', jobId, phase: 'loading-model', progress: fraction });
        }
      }).then(
        (engine) => {
          this.engine = engine;
          this.emit({ type: 'status', status: { state: 'ready', model: summarize(engine.manifest), runningOn: engine.runningOn } });
          return engine;
        },
        (error: unknown) => {
          this.loading = null;
          if (error instanceof EchoError && error.code === 'MODEL_NOT_DOWNLOADED') {
            // Not a fault of the engine: there is simply nothing to load yet.
            this.emit({ type: 'status', status: { state: 'unloaded' } });
            throw error;
          }
          const message = error instanceof Error ? error.message : String(error);
          this.emit({ type: 'status', status: { state: 'error', message } });
          throw new EchoError('MODEL_LOAD_FAILED', { message });
        },
      );
    }
    return this.loading;
  }

  warmUp(): void {
    this.ensureEngine().catch(() => {
      /* already reported as an error status */
    });
  }

  enqueue(jobId: string, style: StyleId, text: string): void {
    const controller = new AbortController();
    this.controllers.set(jobId, controller);
    this.queue = this.queue.then(() => this.run(jobId, style, text, controller.signal));
  }

  cancel(jobId: string): void {
    this.controllers.get(jobId)?.abort();
  }

  /** Resolves when every job queued so far has finished. */
  idle(): Promise<void> {
    return this.queue;
  }

  /**
   * Let a running load and the queued jobs finish, then free the model's
   * memory. The next job loads it again.
   */
  async unload(): Promise<void> {
    await this.loading?.catch(() => undefined);
    await this.queue;
    const engine = this.engine;
    if (!engine) return;
    this.engine = null;
    this.loading = null;
    await engine.release();
    this.emit({ type: 'status', status: { state: 'unloaded' } });
  }

  private async run(jobId: string, style: StyleId, text: string, signal: AbortSignal): Promise<void> {
    try {
      let result: RewriteResult;
      try {
        result = await this.attempt(jobId, style, text, signal);
      } catch (error) {
        if (!(error instanceof EngineBrokenError)) throw error;
        result = await this.attempt(jobId, style, text, signal);
      }
      this.emit({ type: 'job-done', jobId, text: result.text });
    } catch (error) {
      this.emit({ type: 'job-failed', jobId, error: toErrorPayload(error instanceof EngineBrokenError ? error.reason : error, 'INFERENCE_FAILED') });
    } finally {
      this.controllers.delete(jobId);
    }
  }

  private async attempt(jobId: string, style: StyleId, text: string, signal: AbortSignal): Promise<RewriteResult> {
    if (signal.aborted) throw new EchoError('CANCELLED');
    let engine = this.engine;
    if (!engine) {
      this.waitingForModel.add(jobId);
      this.emit({ type: 'job-phase', jobId, phase: 'loading-model', progress: 0 });
      try {
        engine = await this.ensureEngine();
      } finally {
        this.waitingForModel.delete(jobId);
      }
    }
    if (signal.aborted) throw new EchoError('CANCELLED');
    this.emit({ type: 'job-phase', jobId, phase: 'rewriting' });
    try {
      return await engine.rewrite(style, text, { signal });
    } catch (error) {
      if (error instanceof EngineBrokenError) this.discard(engine);
      throw error;
    }
  }

  /** Forget a broken engine; the next job loads a new one. Releasing it is not waited on: it must not hold the job up. */
  private discard(engine: LoadedEngine): void {
    if (this.engine === engine) {
      this.engine = null;
      this.loading = null;
    }
    engine.release().catch(() => undefined);
  }
}
