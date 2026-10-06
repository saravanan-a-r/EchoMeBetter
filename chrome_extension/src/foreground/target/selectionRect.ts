/**
 * Where the selected text of a <textarea> or <input> is on screen.
 *
 * The browser only reports a text control's own box, never the selection
 * inside it, so an anchor taken from the element lands at the bottom of a
 * tall textarea even when the selection is its first two lines. The text is
 * laid out again in an invisible copy of the control (same box, same font,
 * same wrapping) and the selection is measured there.
 */

export interface Rect {
  readonly top: number;
  readonly left: number;
  readonly bottom: number;
  readonly right: number;
  readonly width: number;
  readonly height: number;
}

export function rectOf(top: number, left: number, bottom: number, right: number): Rect {
  return { top, left, bottom, right, width: right - left, height: bottom - top };
}

/** Style properties that decide where a character falls. */
const LAYOUT_PROPERTIES = [
  'direction', 'writingMode', 'textOrientation',
  'fontFamily', 'fontSize', 'fontStyle', 'fontWeight', 'fontStretch', 'fontVariant', 'fontKerning', 'fontFeatureSettings',
  'lineHeight', 'letterSpacing', 'wordSpacing', 'textTransform', 'textIndent', 'textAlign', 'textRendering', 'tabSize',
  'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
  'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth',
  'borderTopStyle', 'borderRightStyle', 'borderBottomStyle', 'borderLeftStyle',
] as const;

/**
 * The selection's box relative to the control's own top-left corner, as laid
 * out in the invisible copy (the copy sits at the document's origin, so what
 * is measured is already relative).
 */
function measureInCopy(element: HTMLInputElement | HTMLTextAreaElement, start: number, end: number): Rect | null {
  const doc = element.ownerDocument;
  const view = doc.defaultView;
  if (!view || !doc.documentElement) return null;
  const computed = view.getComputedStyle(element);
  const isInput = element.tagName === 'INPUT';

  const holder = doc.createElement('div');
  const shadow = holder.attachShadow({ mode: 'closed' });
  const copy = doc.createElement('div');
  for (const property of LAYOUT_PROPERTIES) copy.style[property] = computed[property];
  const scrollbar = Math.max(0, element.offsetWidth - element.clientWidth - parseFloat(computed.borderLeftWidth) - parseFloat(computed.borderRightWidth));
  Object.assign(copy.style, {
    position: 'absolute',
    top: '0',
    left: '0',
    boxSizing: 'border-box',
    width: `${element.offsetWidth}px`,
    height: 'auto',
    margin: '0',
    overflow: 'hidden',
    visibility: 'hidden',
    pointerEvents: 'none',
    // A textarea wraps like pre-wrap; an input never wraps.
    whiteSpace: isInput ? 'pre' : 'pre-wrap',
    overflowWrap: isInput ? 'normal' : 'break-word',
    // The space a scrollbar takes comes out of the line width.
    paddingRight: `${parseFloat(computed.paddingRight) + scrollbar}px`,
  });
  copy.textContent = element.value;
  shadow.append(copy);
  // Anchored to the document's root so the page's own layout around the control cannot move it.
  Object.assign(holder.style, { position: 'absolute', top: '0', left: '0', width: '0', height: '0', overflow: 'visible' });
  doc.documentElement.append(holder);
  try {
    const text = copy.firstChild;
    if (!text) return null;
    const range = doc.createRange();
    range.setStart(text, Math.min(start, element.value.length));
    range.setEnd(text, Math.min(end, element.value.length));
    const box = range.getBoundingClientRect();
    const origin = copy.getBoundingClientRect();
    if (box.width === 0 && box.height === 0) return null;
    return rectOf(box.top - origin.top, box.left - origin.left, box.bottom - origin.top, box.right - origin.left);
  } finally {
    holder.remove();
  }
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

/** `inner` limited to `outer`; a rect wholly outside collapses onto the nearest edge. */
function limitTo(inner: Rect, outer: Rect): Rect {
  return rectOf(
    clamp(inner.top, outer.top, outer.bottom),
    clamp(inner.left, outer.left, outer.right),
    clamp(inner.bottom, outer.top, outer.bottom),
    clamp(inner.right, outer.left, outer.right),
  );
}

export interface Placement {
  /** The control's border box on screen. */
  readonly element: Rect;
  /** The selection measured in the copy, relative to the control's corner. */
  readonly inCopy: Rect;
  readonly scrollTop: number;
  readonly scrollLeft: number;
  readonly singleLine: boolean;
}

/**
 * The selection's box on screen: the copy's position moved to where the
 * control is and shifted by how far the control has scrolled, then limited to
 * the part of the control that is visible. A selection scrolled out of view
 * is pinned to the nearest edge of the control.
 */
export function placeSelection({ element, inCopy, scrollTop, scrollLeft, singleLine }: Placement): Rect {
  const left = element.left + inCopy.left - scrollLeft;
  const right = element.left + inCopy.right - scrollLeft;
  // An input centres its one line vertically, which the copy does not model.
  const top = singleLine ? element.top : element.top + inCopy.top - scrollTop;
  const bottom = singleLine ? element.bottom : element.top + inCopy.bottom - scrollTop;
  return limitTo(rectOf(top, left, bottom, right), element);
}

export function textControlSelectionRect(element: HTMLInputElement | HTMLTextAreaElement, start: number, end: number): Rect {
  const box = element.getBoundingClientRect();
  const element_ = rectOf(box.top, box.left, box.bottom, box.right);
  try {
    const inCopy = measureInCopy(element, start, end);
    if (!inCopy) return element_;
    return placeSelection({ element: element_, inCopy, scrollTop: element.scrollTop, scrollLeft: element.scrollLeft, singleLine: element.tagName === 'INPUT' });
  } catch {
    // Anything unexpected about the page's styles: the control's own box is still a usable anchor.
    return element_;
  }
}
