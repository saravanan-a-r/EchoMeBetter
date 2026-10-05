/**
 * Mount the overlay into an isolated shadow root on the page.
 *
 * Isolation from the host page, layer by layer:
 *   - JavaScript: the content script runs in Chrome's isolated world, with
 *     its own globals and its own bundled React; the page cannot see or
 *     clash with either.
 *   - CSS into the overlay: page selectors do not cross the shadow
 *     boundary, and `.echomebetter-root { all: initial }` stops inheritance.
 *   - CSS onto the host element: page rules like `* { display: none !important }`
 *     still match the host itself, so its own styles are set `!important`
 *     inline -- the one thing no stylesheet can override.
 *   - Stacking: the host is a manual popover, i.e. in the browser's top
 *     layer. No page z-index, `overflow: hidden` or transformed ancestor can
 *     hide or offset it.
 *   - Events: clicks and keys inside the overlay are stopped at its root so
 *     page "click outside" handlers (closing a compose window, a menu) never
 *     see them.
 * Mounting is lazy -- nothing is added to the page until the first job.
 */
import { createRoot, type Root } from 'react-dom/client';
import { adoptStyles } from './adoptStyles';
import overlayCss from './overlay.css?inline';
import { OverlayApp } from './OverlayApp';
import { OverlayStore } from './overlayStore';

export const OVERLAY_TAG = 'echomebetter-overlay';

export interface Overlay {
  readonly store: OverlayStore;
  readonly host: HTMLElement;
  unmount(): void;
}

/** Host styles, all `!important`: inline important declarations beat every page stylesheet. */
const HOST_STYLES: Record<string, string> = {
  all: 'initial',
  display: 'block',
  position: 'fixed',
  inset: 'auto',
  top: '0',
  left: '0',
  width: '0',
  height: '0',
  margin: '0',
  padding: '0',
  border: '0',
  background: 'transparent',
  overflow: 'visible',
  'z-index': '2147483647',
  'pointer-events': 'none',
};

/** Events that must not reach the page's own listeners. */
const CONTAINED_EVENTS = ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'click', 'dblclick', 'contextmenu', 'keydown', 'keyup', 'focusin', 'focusout'];

function showInTopLayer(host: HTMLElement): void {
  // The popover API is in every Chrome the extension supports (116+);
  // the check only keeps non-browser test environments working.
  if (typeof host.showPopover !== 'function') return;
  if (host.matches(':popover-open')) host.hidePopover();
  host.showPopover();
}

export function mountOverlay(doc: Document = document): Overlay {
  const host = doc.createElement(OVERLAY_TAG);
  host.setAttribute('popover', 'manual');
  // CSSOM, not a style attribute: a page's CSP may forbid inline styles.
  for (const [property, value] of Object.entries(HOST_STYLES)) host.style.setProperty(property, value, 'important');
  const shadow = host.attachShadow({ mode: 'open' });
  adoptStyles(shadow, overlayCss);
  const container = doc.createElement('div');
  container.className = 'echomebetter-root';
  shadow.append(container);
  doc.documentElement.appendChild(host);
  showInTopLayer(host);

  const store = new OverlayStore();
  const root: Root = createRoot(container);
  root.render(<OverlayApp store={store} />);
  // Something new to show (a job starting, a toast) re-enters the top layer,
  // so it lands above anything the page opened in the meantime.
  let shown = store.getSnapshot();
  store.subscribe(() => {
    const next = store.getSnapshot();
    if ((next.working && !shown.working) || (next.toast && next.toast.id !== shown.toast?.id)) showInTopLayer(host);
    shown = next;
  });
  // Registered after createRoot, so React's own root listeners still run first.
  for (const type of CONTAINED_EVENTS) container.addEventListener(type, (event) => event.stopPropagation());

  return {
    store,
    host,
    unmount() {
      root.unmount();
      host.remove();
    },
  };
}
