/**
 * One rewrite, from the page's point of view.
 *
 *   start   capture the selection → show the pointer loader → ask for the rewrite
 *   events  "loading model 40%" / "rewriting" update the loader
 *   done    re-check the text is untouched → replace it → toast with Undo
 *           (and, when the model first had to wake up from an idle rest,
 *           how long that took and a way to keep it awake longer)
 *   Esc     cancel; the page is left exactly as it was
 *
 * It also owns the style menu that a press and hold on selected text opens:
 * a pick re-selects the text the menu was opened for and asks for that
 * style the way a shortcut does; Esc or a click elsewhere closes it.
 *
 * Only one job runs per frame. Everything the controller touches outside
 * itself (port, overlay, clipboard) is injected, so the whole journey is
 * testable in a DOM without Chrome.
 */
import { describeError, EchoError, toErrorPayload, type ErrorPayload } from '../shared/errors';
import { formatDuration } from '../shared/format';
import { isJobEvent, type ForegroundMessage, type JobEvent, type JobRequest, type Point, type StartJobReply, type WokeFromRest } from '../shared/messages';
import type { SettingsFocus } from '../shared/popupIntent';
import { keepLoadedLabel } from '../shared/settings';
import { STYLE_IDS, styleLabel, type StyleId } from '../shared/styles';
import { takeHeldSelection } from './heldSelection';
import { applyRewrite, isUnchanged, reselect } from './target/apply';
import { captureTarget, splitWhitespace, targetRect, type EditTarget } from './target/capture';
import { setBusyCursor } from './ui/busyCursor';
import { OVERLAY_TAG } from './ui/mountOverlay';
import type { AnchorRect, OverlayStore, ToastAction, ToastNote, ToastState } from './ui/overlayStore';

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
  /** Open the toolbar popup's settings at `focus`. */
  readonly openSettings: (focus: SettingsFocus) => void;
  /** A style was picked from the style menu: ask for it, as its shortcut would. */
  readonly requestRewrite: (style: StyleId) => void;
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

interface OpenMenu {
  /** The selection the menu was opened for. */
  readonly target: EditTarget;
  readonly stopListening: () => void;
}

export const TOAST_MS = { success: 6000, info: 4000, error: 7000, withNote: 12000 } as const;

export const KEEP_AWAKE_LABEL = 'Keep it awake longer';

function toAnchor(rect: Pick<DOMRect, 'top' | 'left' | 'bottom' | 'right'>): AnchorRect {
  return { top: rect.top, left: rect.left, bottom: rect.bottom, right: rect.right };
}

export class ForegroundController {
  private active: ActiveJob | null = null;
  private menu: OpenMenu | null = null;

  constructor(private readonly deps: ForegroundDeps) {}

  get busy(): boolean {
    return this.active !== null;
  }

  handleMessage(message: ForegroundMessage): StartJobReply {
    if (message.kind === 'echo/notice') {
      this.toastError(message.error, null);
      return { ok: true };
    }
    if (message.kind === 'echo/style-menu') return this.openMenu(message.point);
    return this.start(message.jobId, message.style);
  }

  get menuOpen(): boolean {
    return this.menu !== null;
  }

  /** The style menu, beside `point`, for the text held there. */
  openMenu(point: Point): StartJobReply {
    this.closeMenu();
    if (this.active) return this.refuse(new EchoError('BUSY'), this.active.anchor);
    let target = takeHeldSelection(this.deps.win, point);
    if (!target) {
      try {
        target = captureTarget(this.deps.doc);
      } catch (error) {
        return { ok: false, error: toErrorPayload(error) }; // nothing selected any more: nothing to offer
      }
    }
    if (!isUnchanged(target)) return { ok: false, error: { code: 'TEXT_CHANGED' } };
    const held = target;
    // Letting go of the hold can drop the selection (the browser's or the editor's doing): show it again.
    reselect(held);

    const { win } = this.deps;
    const overlay = () => this.deps.overlay();
    const move = (by: number) => {
      const menu = overlay().getSnapshot().menu;
      if (menu) overlay().setMenuActive((menu.active + by + menu.styles.length) % menu.styles.length);
    };
    const onKey = (event: KeyboardEvent) => {
      const menu = overlay().getSnapshot().menu;
      if (!menu || event.isComposing) return;
      const plain = !event.altKey && !event.ctrlKey && !event.metaKey;
      const digit = plain && /^[1-9]$/.test(event.key) ? Number(event.key) - 1 : -1;
      if (event.key === 'Escape') this.closeMenu();
      else if (event.key === 'ArrowDown' || (event.key === 'Tab' && !event.shiftKey)) move(1);
      else if (event.key === 'ArrowUp' || (event.key === 'Tab' && event.shiftKey)) move(-1);
      else if (event.key === 'Home') overlay().setMenuActive(0);
      else if (event.key === 'End') overlay().setMenuActive(menu.styles.length - 1);
      else if (event.key === 'Enter') this.pick(menu.styles[menu.active]!);
      else if (digit >= 0 && digit < menu.styles.length) this.pick(menu.styles[digit]!);
      else return; // anything else is the page's
      // The menu's keys must not reach the editor (an arrow would move the caret, Enter add a line) or the page.
      event.preventDefault();
      event.stopImmediatePropagation();
    };
    const onPointerDown = (event: PointerEvent) => {
      const inMenu = event.composedPath().some((node) => node instanceof Element && node.localName === OVERLAY_TAG);
      if (!inMenu) this.closeMenu();
    };
    let reselectTimer: number | undefined;
    const onRelease = () => {
      // After the release's own default action (and the editor's handlers) have run.
      reselectTimer = win.setTimeout(() => {
        if (this.menu?.target === held && isUnchanged(held)) reselect(held);
      }, 0);
    };
    win.addEventListener('keydown', onKey, true);
    win.addEventListener('pointerdown', onPointerDown, true);
    win.addEventListener('pointerup', onRelease, { capture: true, once: true });

    this.menu = {
      target: held,
      stopListening: () => {
        win.removeEventListener('keydown', onKey, true);
        win.removeEventListener('pointerdown', onPointerDown, true);
        win.removeEventListener('pointerup', onRelease, true);
        win.clearTimeout(reselectTimer);
      },
    };
    overlay().openMenu({ point, styles: STYLE_IDS, active: 0, pick: (style) => this.pick(style), setActive: (index) => overlay().setMenuActive(index) });
    return { ok: true };
  }

  closeMenu(): void {
    const menu = this.menu;
    if (!menu) return;
    this.menu = null;
    menu.stopListening();
    this.deps.overlay().closeMenu();
  }

  private pick(style: StyleId): void {
    const menu = this.menu;
    if (!menu) return;
    this.closeMenu();
    // Typed over meanwhile: what the menu was opened for is gone.
    if (!isUnchanged(menu.target)) return;
    reselect(menu.target);
    this.deps.requestRewrite(style);
  }

  start(jobId: string, style: StyleId): StartJobReply {
    this.closeMenu();
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
        this.deliver(job, event.text, event.wokeFromRest);
        return;
      case 'job/failed':
        this.fail(event.error);
        return;
    }
  }

  private now(): number {
    return (this.deps.now ?? (() => performance.now()))();
  }

  /** How long the model took to wake up, and that a longer keep-awake time would skip it. */
  private wakeNote(job: ActiveJob, rest: WokeFromRest | undefined): ToastNote | undefined {
    if (!rest || job.rewriteStartedAt === null) return undefined;
    const waking = formatDuration(job.rewriteStartedAt - job.requestedAt);
    return {
      text: `Waking up took ${waking}. EchoMeBetter rests after ${keepLoadedLabel(rest.idleMinutes)} without use, to keep your computer fast.`,
      action: { label: KEEP_AWAKE_LABEL, run: () => this.deps.openSettings('keep-awake') },
    };
  }

  private deliver(job: ActiveJob, text: string, rest?: WokeFromRest): void {
    const took = formatDuration(this.now() - (job.rewriteStartedAt ?? job.requestedAt));
    const note = this.wakeNote(job, rest);
    if (text === job.core) {
      this.toast({
        tone: 'info',
        title: 'Looks good already',
        message: `No ${styleLabel(job.style).toLowerCase()} changes to suggest. Took ${took}.`,
        actions: [],
        note,
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
        note,
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
    // A note is more to read; hovering the toast keeps it up as well.
    this.deps.overlay().showToast({ ...toast, durationMs: toast.note ? TOAST_MS.withNote : TOAST_MS[toast.tone] });
  }
}
