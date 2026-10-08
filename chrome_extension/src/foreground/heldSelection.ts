/**
 * The selection a press and hold was made on, handed from the page listener
 * (shortcuts.js, always on the page) to the style menu (foreground.js,
 * injected when the hold completes).
 *
 * Both scripts run in the frame's one content-script world, so they share
 * its window; the selection is left there under a registry symbol, the way
 * each script already marks itself installed. A selection is only taken by
 * the menu opened for the same hold (the same point), and only once.
 */
import type { EditTarget } from './target/capture';
import type { Point } from './ui/overlayStore';

const HELD = Symbol.for('echomebetter.heldSelection');

interface Held {
  readonly point: Point;
  readonly target: EditTarget;
}

type Registry = Record<symbol, Held | undefined>;

export function leaveHeldSelection(win: Window, point: Point, target: EditTarget): void {
  (win as unknown as Registry)[HELD] = { point, target };
}

export function takeHeldSelection(win: Window, point: Point): EditTarget | null {
  const registry = win as unknown as Registry;
  const held = registry[HELD];
  registry[HELD] = undefined;
  return held && held.point.x === point.x && held.point.y === point.y ? held.target : null;
}
