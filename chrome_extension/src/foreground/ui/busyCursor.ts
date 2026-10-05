/**
 * Turns the page's mouse pointer into EchoMeBetter's "working" pointer while
 * a rewrite runs, and back afterwards.
 *
 * One stylesheet and one attribute on <html>; both are removed when the job
 * ends, leaving the page exactly as it was.
 */
import { adoptStyles } from './adoptStyles';

const ATTRIBUTE = 'data-echomebetter-busy';
const removers = new WeakMap<Document, () => void>();

// A brand-violet arrow with a mint dot, falling back to the system "progress" cursor.
const CURSOR_SVG =
  "<svg xmlns='http://www.w3.org/2000/svg' width='28' height='28' viewBox='0 0 28 28'>" +
  "<path d='M4 3 L4 21 L9 16.5 L12.5 24 L15.5 22.7 L12 15.3 L18.5 15 Z' fill='%237650FF' stroke='white' stroke-width='1.6' stroke-linejoin='round'/>" +
  "<circle cx='21' cy='21' r='4' fill='%232DD4BF' stroke='white' stroke-width='1.5'/></svg>";

const CSS = `[${ATTRIBUTE}], [${ATTRIBUTE}] * { cursor: url("data:image/svg+xml;utf8,${CURSOR_SVG}") 4 3, progress !important; }`;

export function setBusyCursor(doc: Document, busy: boolean): void {
  const root = doc.documentElement;
  if (busy) {
    if (!removers.has(doc)) removers.set(doc, adoptStyles(doc, CSS));
    root.setAttribute(ATTRIBUTE, '');
  } else {
    root.removeAttribute(ATTRIBUTE);
    removers.get(doc)?.();
    removers.delete(doc);
  }
}

export function isBusyCursorOn(doc: Document): boolean {
  return doc.documentElement.hasAttribute(ATTRIBUTE);
}
