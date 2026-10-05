/**
 * One rewrite, from the page's point of view.
 *
 *   start   capture the selection → show the pointer loader → ask for the rewrite
 *   events  "loading model 40%" / "rewriting" update the loader
 *   done    re-check the text is untouched → replace it → toast with Undo
 *   Esc     cancel; the page is left exactly as it was
 *
 * Only one job runs per frame. Everything the controller touches outside
 * itself (port, overlay, clipboard) is injected, so the whole journey is
 * testable in a DOM without Chrome.
 */
import { describeError, EchoError, toErrorPayload, type ErrorPayload } from '../shared/errors';
import { isJobEvent, type ForegroundMessage, type JobEvent, type JobRequest, type StartJobReply } from '../shared/messages';
import { formatDuration } from '../shared/format';
import { styleLabel, type StyleId } from '../shared/styles';
import { applyRewrite } from './target/apply';
import { captureTarget, splitWhitespace, targetRect, type EditTarget } from './target/capture';
import { setBusyCursor } from './ui/busyCursor';
import type { AnchorRect, OverlayStore, ToastAction, ToastState } from './ui/overlayStore';

export interface JobPortLike {
  postMessage(message: JobRequest): void;
  disconnect(): void;
  onMessage: { addListener(listener: (message: unknown) => void): void };
  onDisconnect: { addListener(listener: () => void): void };
}

export interface ForegroundDeps {
  readonly doc: Document;
  readonly win: Window;
  readonly connect: () => JobPortLike;
  /** The overlay's store, mounting the overlay on first use. */
  readonly overlay: () => OverlayStore;
  readonly writeClipboard: (text: string) => Promise<void>;
  /** Milliseconds on a monotonic clock; defaults to performance.now. */
  readonly now?: () => number;
}

interface ActiveJob {
  readonly jobId: string;
  readonly style: StyleId;
  readonly target: EditTarget;
  readonly lead: string;
  readonly core: string;
  readonly trail: string;
  readonly anchor: AnchorRect;
  readonly port: JobPortLike;
  /** When the model started rewriting (after any loading); null until then. */
  rewriteStartedAt: number | null;
  readonly requestedAt: number;
  readonly stopListening: () => void;
}

export const TOAST_MS = { success: 6000, info: 4000, error: 7000 } as const;

function toAnchor(rect: DOMRect): AnchorRect {
  return { top: rect.top, left: rect.left, bottom: rect.bottom, right: rect.right };
}

export class ForegroundController {
  private active: ActiveJob | null = null;

  constructor(private readonly deps: ForegroundDeps) {}

  get busy(): boolean {
    return this.active !== null;
  }

  handleMessage(message: ForegroundMessage): StartJobReply {
    if (message.kind === 'echo/notice') {
      this.toastError(message.error, null);
      return { ok: true };
    }
    return this.start(message.jobId, message.style);
  }

  start(jobId: string, style: StyleId): StartJobReply {
    if (this.active) return this.refuse(new EchoError('BUSY'), this.active.anchor);

    let target: EditTarget;
    try {
      target = captureTarget(this.deps.doc);
    } catch (error) {
      const active = this.deps.doc.activeElement;
      return this.refuse(error, active ? toAnchor(active.getBoundingClientRect()) : null);
    }

    const { lead, core, trail } = splitWhitespace(target.text);
    const rect = targetRect(target);
    const anchor = toAnchor(rect);
    let port: JobPortLike;
    try {
      port = this.deps.connect();
    } catch (error) {
      // The extension was reloaded or updated under this page.
      return this.refuse(new EchoError('DISCONNECTED', { message: error instanceof Error ? error.message : String(error) }), anchor);
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      this.cancel();
    };
    this.deps.win.addEventListener('keydown', onKey, true);

    this.active = {
      jobId,
      style,
      target,
      lead,
      core,
      trail,
      anchor,
      port,
      rewriteStartedAt: null,
      requestedAt: this.now(),
      stopListening: () => this.deps.win.removeEventListener('keydown', onKey, true),
    };

    port.onMessage.addListener((message) => {
      if (isJobEvent(message) && message.jobId === jobId) this.onEvent(message);
    });
    port.onDisconnect.addListener(() => {
      if (this.active?.port === port) this.fail({ code: 'DISCONNECTED' });
    });

    this.deps.overlay().startWorking({
      style,
      phase: 'starting',
      origin: {
        x: Math.min(Math.max(rect.left + Math.min(rect.width, 240) / 2, 0), this.deps.win.innerWidth),
        y: Math.min(Math.max(rect.top + Math.min(rect.height, 48) / 2, 0), this.deps.win.innerHeight),
      },
    });
    setBusyCursor(this.deps.doc, true);
    port.postMessage({ kind: 'job/request', jobId, style, text: core });
    return { ok: true };
  }

  cancel(): void {
    const job = this.active;
    if (!job) return;
    try {
      job.port.postMessage({ kind: 'job/cancel', jobId: job.jobId });
    } catch {
      // Port already gone: nothing is running for this job anymore.
    }
    this.end();
    this.toast({ tone: 'info', title: 'Rewrite cancelled', message: 'Your text was left unchanged.', actions: [], anchor: job.anchor });
  }

  private onEvent(event: JobEvent): void {
    const job = this.active;
    if (!job) return;
    switch (event.kind) {
      case 'job/phase':
        if (event.phase === 'rewriting' && job.rewriteStartedAt === null) job.rewriteStartedAt = this.now();
        this.deps.overlay().updateWorking({ phase: event.phase, progress: event.progress });
        return;
      case 'job/done':
        this.end();
        this.deliver(job, event.text);
        return;
      case 'job/failed':
        this.fail(event.error);
        return;
    }
  }

  private now(): number {
    return (this.deps.now ?? (() => performance.now()))();
  }

  private deliver(job: ActiveJob, text: string): void {
    const took = formatDuration(this.now() - (job.rewriteStartedAt ?? job.requestedAt));
    if (text === job.core) {
      this.toast({
        tone: 'info',
        title: 'Looks good already',
        message: `No ${styleLabel(job.style).toLowerCase()} changes to suggest. Took ${took}.`,
        actions: [],
        anchor: job.anchor,
      });
      return;
    }
    try {
      const edit = applyRewrite(job.target, `${job.lead}${text}${job.trail}`);
      this.toast({
        tone: 'success',
        title: `Rewritten · ${styleLabel(job.style)}`,
        message: `Took ${took}. Not quite right? Undo restores your original.`,
        actions: [
          {
            label: 'Undo',
            run: () => {
              if (edit.undo()) {
                this.toast({ tone: 'info', title: 'Original text restored', actions: [], anchor: job.anchor });
              } else {
                this.toast({
                  tone: 'error',
                  title: "Couldn't undo",
                  message: 'The text has been edited since. Use Ctrl/⌘+Z in the field instead.',
                  actions: [],
                  anchor: job.anchor,
                });
              }
            },
          },
        ],
        anchor: job.anchor,
      });
    } catch (error) {
      const payload = toErrorPayload(error);
      const copy: ToastAction = {
        label: 'Copy rewrite',
        run: () => {
          this.deps.writeClipboard(text).catch(() => undefined);
        },
      };
      this.toastError(payload, job.anchor, payload.code === 'TEXT_CHANGED' ? [copy] : []);
    }
  }

  private fail(error: ErrorPayload): void {
    const job = this.active;
    if (!job) return;
    this.end();
    if (error.code !== 'CANCELLED') this.toastError(error, job.anchor);
  }

  private end(): void {
    const job = this.active;
    if (!job) return;
    this.active = null;
    job.stopListening();
    this.deps.overlay().stopWorking();
    setBusyCursor(this.deps.doc, false);
    try {
      job.port.disconnect();
    } catch {
      // Already disconnected.
    }
  }

  private refuse(error: unknown, anchor: AnchorRect | null): StartJobReply {
    const payload = toErrorPayload(error);
    this.toastError(payload, anchor);
    return { ok: false, error: payload };
  }

  private toastError(error: ErrorPayload, anchor: AnchorRect | null, actions: ToastAction[] = []): void {
    const gentle = error.code === 'NO_SELECTION' || error.code === 'BUSY';
    this.toast({
      tone: gentle ? 'info' : 'error',
      title: error.code === 'NO_SELECTION' ? 'Nothing selected' : error.code === 'BUSY' ? 'One at a time' : "Couldn't rewrite that",
      message: describeError(error),
      actions,
      anchor,
    });
  }

  private toast(toast: Omit<ToastState, 'id' | 'durationMs'>): void {
    this.deps.overlay().showToast({ ...toast, durationMs: TOAST_MS[toast.tone] });
  }
}
