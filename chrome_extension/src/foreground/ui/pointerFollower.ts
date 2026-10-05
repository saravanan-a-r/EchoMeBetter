/**
 * Keeps an element next to the mouse pointer.
 *
 * Built for zero impact on the page: one passive listener, at most one DOM
 * write per animation frame, and that write is a `transform` (compositor
 * only, no layout). React never re-renders on pointer movement.
 */
import type { Point } from './overlayStore';

export const POINTER_OFFSET = { x: 18, y: 20 } as const;

export function followPointer(element: HTMLElement, origin: Point, win: Window = window): () => void {
  let target = origin;
  let frame = 0;
  // The element's size is cached and kept current by a ResizeObserver, so
  // following the pointer never forces the page to recalculate layout.
  let { width, height } = element.getBoundingClientRect();
  const resize = typeof ResizeObserver === 'function'
    ? new ResizeObserver(([entry]) => {
        if (!entry) return;
        const box = entry.borderBoxSize?.[0];
        width = box ? box.inlineSize : entry.contentRect.width;
        height = box ? box.blockSize : entry.contentRect.height;
        if (!frame) frame = win.requestAnimationFrame(place);
      })
    : null;

  const place = () => {
    frame = 0;
    const { innerWidth, innerHeight } = win;
    // Stay on screen: flip to the other side of the pointer near the edges.
    const x = target.x + POINTER_OFFSET.x + width > innerWidth ? target.x - POINTER_OFFSET.x - width : target.x + POINTER_OFFSET.x;
    const y = target.y + POINTER_OFFSET.y + height > innerHeight ? target.y - POINTER_OFFSET.y - height : target.y + POINTER_OFFSET.y;
    element.style.transform = `translate3d(${Math.max(4, x)}px, ${Math.max(4, y)}px, 0)`;
  };

  const onMove = (event: PointerEvent) => {
    target = { x: event.clientX, y: event.clientY };
    if (!frame) frame = win.requestAnimationFrame(place);
  };

  place();
  resize?.observe(element);
  win.addEventListener('pointermove', onMove, { passive: true, capture: true });
  return () => {
    resize?.disconnect();
    win.removeEventListener('pointermove', onMove, { capture: true });
    if (frame) win.cancelAnimationFrame(frame);
  };
}
