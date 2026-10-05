/**
 * Find what the user selected for rewriting, and remember exactly where it is.
 *
 * Two kinds of editable target:
 *   - text controls (<textarea>, text-like <input>): an offset range into `value`
 *   - contenteditable regions: a live DOM Range inside the editing host
 *
 * The snapshot taken here is what `apply.ts` later checks against, so text
 * that changed while the model was working is never overwritten.
 */
import { EchoError } from '../../shared/errors';

/** Input types whose selection can be read and replaced (the HTML selection APIs apply to these). */
export const TEXT_INPUT_TYPES = new Set(['text', 'search', 'url', 'tel']);

export interface TextControlTarget {
  readonly kind: 'text-control';
  readonly element: HTMLInputElement | HTMLTextAreaElement;
  readonly start: number;
  readonly end: number;
  readonly text: string;
}

export interface ContentEditableTarget {
  readonly kind: 'content-editable';
  readonly host: HTMLElement;
  /** Live range: the browser keeps it pointing at the selected content as the DOM changes. */
  readonly range: Range;
  /** The selection as the user sees it (line breaks included), sent to the model. */
  readonly text: string;
  /** `range.toString()` at capture time, for the unchanged-check. */
  readonly rangeText: string;
  readonly root: Document | ShadowRoot;
}

export type EditTarget = TextControlTarget | ContentEditableTarget;

/** The focused element, looking inside open shadow roots. */
export function deepActiveElement(doc: Document): Element | null {
  let active: Element | null = doc.activeElement;
  while (active?.shadowRoot?.activeElement) active = active.shadowRoot.activeElement;
  return active;
}

function isTextArea(element: Element): element is HTMLTextAreaElement {
  return element.tagName === 'TEXTAREA';
}

function isInput(element: Element): element is HTMLInputElement {
  return element.tagName === 'INPUT';
}

function captureTextControl(element: HTMLInputElement | HTMLTextAreaElement): TextControlTarget {
  if (isInput(element)) {
    const type = (element.getAttribute('type') ?? 'text').toLowerCase();
    if (type === 'password') throw new EchoError('PASSWORD_FIELD');
    if (!TEXT_INPUT_TYPES.has(type)) throw new EchoError('UNSUPPORTED_FIELD', { type });
  }
  if (element.readOnly || element.disabled) throw new EchoError('UNSUPPORTED_FIELD', { reason: 'read-only' });
  const start = element.selectionStart;
  const end = element.selectionEnd;
  if (start === null || end === null) throw new EchoError('UNSUPPORTED_FIELD');
  const text = element.value.slice(start, end);
  if (text.trim().length === 0) throw new EchoError('NO_SELECTION');
  return { kind: 'text-control', element, start, end, text };
}

/** The outermost contenteditable element containing `node` (the editing host). */
export function editingHost(node: Node | null): HTMLElement | null {
  let element: Element | null = node instanceof Element ? node : (node?.parentElement ?? null);
  let host: HTMLElement | null = null;
  while (element) {
    if (element instanceof HTMLElement && element.isContentEditable) host = element;
    else if (host) break;
    element = element.parentElement;
  }
  return host;
}

export function selectionFor(root: Document | ShadowRoot, doc: Document): Selection | null {
  // Chromium exposes the selection inside a shadow tree on the root itself.
  const shadowSelection = (root as ShadowRoot & { getSelection?: () => Selection | null }).getSelection;
  if (root !== doc && typeof shadowSelection === 'function') return shadowSelection.call(root);
  return doc.getSelection();
}

function captureContentEditable(active: Element | null, doc: Document): ContentEditableTarget {
  const root = (active?.getRootNode() as Document | ShadowRoot | undefined) ?? doc;
  const selection = selectionFor(root, doc);
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) throw new EchoError('NO_SELECTION');
  const range = selection.getRangeAt(0);
  const host = editingHost(range.startContainer);
  if (!host || editingHost(range.endContainer) !== host) throw new EchoError('UNSUPPORTED_FIELD', { reason: 'selection spans outside the editor' });
  const text = selection.toString();
  if (text.trim().length === 0) throw new EchoError('NO_SELECTION');
  return { kind: 'content-editable', host, range: range.cloneRange(), text, rangeText: range.toString(), root };
}

export function captureTarget(doc: Document = document): EditTarget {
  const active = deepActiveElement(doc);
  if (active && (isTextArea(active) || isInput(active))) return captureTextControl(active);
  if (active instanceof HTMLElement && active.isContentEditable) return captureContentEditable(active, doc);
  // Focus can sit on the page while the selection is inside an editor.
  const selection = doc.getSelection();
  if (selection && selection.rangeCount > 0 && editingHost(selection.getRangeAt(0).startContainer)) {
    return captureContentEditable(active, doc);
  }
  const focusedSomething = active !== null && active !== doc.body && active !== doc.documentElement;
  throw new EchoError(focusedSomething ? 'UNSUPPORTED_FIELD' : 'NO_SELECTION');
}

/** Bounding box of the selection on screen, for placing the loader and toasts. */
export function targetRect(target: EditTarget): DOMRect {
  if (target.kind === 'content-editable') {
    const rect = target.range.getBoundingClientRect();
    if (rect.width > 0 || rect.height > 0) return rect;
    return target.host.getBoundingClientRect();
  }
  return target.element.getBoundingClientRect();
}

/** Split off leading/trailing whitespace: the model sees the core, the page keeps its spacing. */
export function splitWhitespace(text: string): { readonly lead: string; readonly core: string; readonly trail: string } {
  const lead = /^\s*/u.exec(text)![0];
  const trail = /\s*$/u.exec(text.slice(lead.length))![0];
  return { lead, core: text.slice(lead.length, text.length - trail.length), trail };
}
