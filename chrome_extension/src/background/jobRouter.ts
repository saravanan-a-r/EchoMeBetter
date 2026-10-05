/**
 * Routes each rewrite job between the page that asked for it and the engine.
 *
 * A page opens one Port per job. The router remembers which port owns which
 * job so engine events find their way back, and treats a port disconnecting
 * (tab closed, navigated, frame removed) as a cancellation: the engine stops
 * spending CPU on text nobody will see.
 */
import { toErrorPayload } from '../shared/errors';
import { isJobRequest, type EngineEvent, type JobEvent, type OffscreenCommand } from '../shared/messages';
import type { StyleId } from '../shared/styles';

export interface PortLike {
  readonly name: string;
  postMessage(message: JobEvent): void;
  onMessage: { addListener(listener: (message: unknown) => void): void };
  onDisconnect: { addListener(listener: () => void): void };
}

export interface RouterDeps {
  /** Make sure the offscreen engine host exists. */
  readonly ensureEngineHost: () => Promise<void>;
  readonly sendToEngine: (command: OffscreenCommand) => Promise<void>;
  /** Called whenever there is user-driven work, to postpone idle unloading. */
  readonly onActivity: () => void;
}

export class JobRouter {
  private readonly jobs = new Map<string, PortLike>();

  constructor(private readonly deps: RouterDeps) {}

  get activeJobCount(): number {
    return this.jobs.size;
  }

  handlePort(port: PortLike): void {
    const owned = new Set<string>();

    port.onMessage.addListener((message) => {
      if (!isJobRequest(message)) return;
      if (message.kind === 'job/cancel') {
        // Forget the job first: a rewrite still waiting on the engine host
        // will then never be sent, and late events for it are dropped.
        if (owned.has(message.jobId) && this.jobs.get(message.jobId) === port) {
          this.jobs.delete(message.jobId);
          this.cancel(message.jobId);
        }
        return;
      }
      owned.add(message.jobId);
      this.jobs.set(message.jobId, port);
      this.deps.onActivity();
      void this.start(message.jobId, message.style, message.text, port);
    });

    port.onDisconnect.addListener(() => {
      for (const jobId of owned) {
        if (this.jobs.get(jobId) === port) {
          this.jobs.delete(jobId);
          this.cancel(jobId);
        }
      }
    });
  }

  private cancel(jobId: string): void {
    // No engine host means nothing is running; a failed send is not an error.
    this.deps.sendToEngine({ target: 'offscreen', kind: 'engine/cancel', jobId }).catch(() => undefined);
  }

  private async start(jobId: string, style: StyleId, text: string, port: PortLike): Promise<void> {
    try {
      await this.deps.ensureEngineHost();
      // The page may have gone away while the engine host was starting.
      if (this.jobs.get(jobId) !== port) return;
      await this.deps.sendToEngine({ target: 'offscreen', kind: 'engine/rewrite', jobId, style, text });
    } catch (error) {
      this.finish(jobId, { kind: 'job/failed', jobId, error: toErrorPayload(error, 'MODEL_LOAD_FAILED') }, port);
    }
  }

  handleEngineEvent(event: EngineEvent): void {
    switch (event.type) {
      case 'job-phase': {
        const port = this.jobs.get(event.jobId);
        port?.postMessage({ kind: 'job/phase', jobId: event.jobId, phase: event.phase, progress: event.progress });
        return;
      }
      case 'job-done':
        this.finish(event.jobId, { kind: 'job/done', jobId: event.jobId, text: event.text });
        this.deps.onActivity();
        return;
      case 'job-failed':
        this.finish(event.jobId, { kind: 'job/failed', jobId: event.jobId, error: event.error });
        return;
      case 'status':
      case 'installed':
      case 'download':
        return;
    }
  }

  private finish(jobId: string, message: JobEvent, port = this.jobs.get(jobId)): void {
    if (!port || this.jobs.get(jobId) !== port) return;
    this.jobs.delete(jobId);
    try {
      port.postMessage(message);
    } catch {
      // The page went away between the event and now; nothing to tell.
    }
  }
}
