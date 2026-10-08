/**
 * Press and hold on selected text to pick a style: the mouse's way to the
 * styles, without the right-click menu.
 *
 * A hold counts when the main button (mouse or pen) goes down on text
 * selected in an editable field, with no modifier keys, and stays down and
 * still for HOLD_MS. Moving starts the browser's usual drag or selection
 * instead, and letting go earlier is an ordinary click: the page sees every
 * event either way. Only the release that ends a hold has its default action
 * prevented, so it does not drop the selection the menu was opened for.
 *
 * What was selected is handed to the style menu as it opens (see
 * heldSelection.ts), because by the time a style is picked the browser, or
 * the website's own editor, may have moved the selection.
 */
import { hasEditableSelection } from './shortcutListener';
import { captureTarget, targetRect, type EditTarget } from './target/capture';
import type { Point } from './ui/overlayStore';

export const HOLD_MS = 2000;
/** How far the pointer may drift during a hold, in CSS pixels. */
export const HOLD_SLOP = 6;

export interface HoldListenerDeps {
  readonly win: Window;
  readonly enabled: () => boolean;
  /** The hold is complete: open the style menu at `point`, for `target`. */
  readonly open: (point: Point, target: EditTarget) => void;
}

function within(point: Point, rect: { top: number; left: number; bottom: number; right: number }, margin: number): boolean {
  return point.x >= rect.left - margin && point.x <= rect.right + margin && point.y >= rect.top - margin && point.y <= rect.bottom + margin;
}

export function listenForHold({ win, enabled, open }: HoldListenerDeps): () => void {
  let pending: { readonly pointerId: number; readonly point: Point; readonly timer: number } | null = null;

  const cancel = () => {
    if (!pending) return;
    win.clearTimeout(pending.timer);
    pending = null;
    stopWatching();
  };

  const onMove = (event: PointerEvent) => {
    if (!pending || event.pointerId !== pending.pointerId) return;
    if (Math.hypot(event.clientX - pending.point.x, event.clientY - pending.point.y) > HOLD_SLOP) cancel();
  };

  const watched: [string, EventListener][] = [
    ['pointermove', onMove as EventListener],
    ['pointerup', cancel],
    ['pointercancel', cancel],
    ['dragstart', cancel],
    ['contextmenu', cancel],
    // The window losing focus (another app), not an element of the page losing it.
    ['blur', ((event: Event) => event.target === win && cancel()) as EventListener],
  ];
  const stopWatching = () => watched.forEach(([type, listener]) => win.removeEventListener(type, listener, true));

  /** The release that ends a completed hold keeps the selection: its default would collapse it. */
  const keepSelectionOnRelease = () => {
    const swallow = (event: Event) => event.preventDefault();
    const done = () => {
      win.removeEventListener('mouseup', swallow, true);
      win.removeEventListener('pointerup', done, true);
      // The click that follows is the same release.
      win.addEventListener('click', swallow, { capture: true, once: true });
      win.setTimeout(() => win.removeEventListener('click', swallow, true), 0);
    };
    win.addEventListener('mouseup', swallow, true);
    win.addEventListener('pointerup', done, true);
  };

  const complete = (point: Point) => {
    pending = null;
    stopWatching();
    let target: EditTarget;
    try {
      target = captureTarget(win.document);
    } catch {
      return; // the selection went away meanwhile
    }
    // Held on the selected text itself, not just anywhere while something is selected.
    if (!within(point, targetRect(target), 4)) return;
    keepSelectionOnRelease();
    open(point, target);
  };

  const onDown = (event: PointerEvent) => {
    cancel();
    if (!enabled() || event.button !== 0 || !event.isPrimary || event.pointerType === 'touch') return;
    if (event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
    if (!hasEditableSelection(win.document)) return;
    const point = { x: event.clientX, y: event.clientY };
    pending = { pointerId: event.pointerId, point, timer: win.setTimeout(() => complete(point), HOLD_MS) };
    watched.forEach(([type, listener]) => win.addEventListener(type, listener, true));
  };

  // Capture phase and passive: the page's own handlers still run, untouched.
  win.addEventListener('pointerdown', onDown, { capture: true, passive: true });
  return () => {
    cancel();
    win.removeEventListener('pointerdown', onDown, true);
  };
}
