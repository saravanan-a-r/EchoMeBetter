/**
 * Put the rewrite back where the selection was.
 *
 * Preferred path: `document.execCommand('insertText' | 'insertHTML')`. It is
 * the one insertion the browser treats as the user typing: it fires
 * `beforeinput`/`input` so frameworks (React, Vue, ProseMirror, Lexical, the
 * Gmail composer, ...) update their own state, and it lands on the native
 * undo stack, so Ctrl/Cmd+Z restores the original. When an editor refuses it,
 * the text is written directly and an `input` event dispatched.
 *
 * Nothing is written if the selected text changed while the model worked.
 */
import { EchoError } from '../../shared/errors';
import type { ContentEditableTarget, EditTarget, TextControlTarget } from './capture';

export interface AppliedEdit {
  /** Restore the original text; false when the field has changed since and undo is no longer safe. */
  undo(): boolean;
}

export function isUnchanged(target: EditTarget): boolean {
  if (target.kind === 'text-control') {
    const { element, start, end, text } = target;
    return element.isConnected && element.value.slice(start, end) === text;
  }
  const { host, range, rangeText } = target;
  return host.isConnected && host.contains(range.commonAncestorContainer) && range.toString() === rangeText;
}

function execInsertText(doc: Document, text: string): boolean {
  try {
    return doc.execCommand('insertText', false, text);
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// <input> / <textarea>
// ---------------------------------------------------------------------------

function replaceInTextControl(element: HTMLInputElement | HTMLTextAreaElement, start: number, end: number, text: string): void {
  element.focus({ preventScroll: true });
  element.setSelectionRange(start, end);
  const doc = element.ownerDocument;
  if (execInsertText(doc, text) && element.value.slice(start, start + text.length) === text) return;
  // The page vetoed the native edit (or the browser lacks it): write it directly.
  element.setRangeText(text, start, end, 'end');
  element.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertReplacementText', data: text }));
}

function applyToTextControl(target: TextControlTarget, replacement: string): AppliedEdit {
  const { element, start, end, text: original } = target;
  // A single-line <input> silently drops line breaks; make that explicit.
  const text = element.tagName === 'INPUT' ? replacement.replace(/\s*[\r\n]+\s*/g, ' ') : replacement.replace(/\r\n?/g, '\n');
  replaceInTextControl(element, start, end, text);
  return {
    undo() {
      if (!element.isConnected || element.value.slice(start, start + text.length) !== text) return false;
      replaceInTextControl(element, start, start + text.length, original);
      return true;
    },
  };
}

// ---------------------------------------------------------------------------
// contenteditable
// ---------------------------------------------------------------------------

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

function selectRange(target: ContentEditableTarget): void {
  target.host.focus({ preventScroll: true });
  const root = target.root as (Document | ShadowRoot) & { getSelection?: () => Selection | null };
  const selection = root.getSelection?.() ?? target.host.ownerDocument.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(target.range);
}

/** Text as DOM: lines separated by <br>, the way a contenteditable shows line breaks. */
function textFragment(doc: Document, text: string): DocumentFragment {
  const fragment = doc.createDocumentFragment();
  text.split('\n').forEach((line, index) => {
    if (index > 0) fragment.appendChild(doc.createElement('br'));
    if (line.length > 0) fragment.appendChild(doc.createTextNode(line));
  });
  return fragment;
}

function insertDirectly(target: ContentEditableTarget, text: string): Node[] {
  const fragment = textFragment(target.host.ownerDocument, text);
  const nodes = Array.from(fragment.childNodes);
  target.range.deleteContents();
  target.range.insertNode(fragment);
  target.host.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertReplacementText', data: text }));
  return nodes;
}

function applyToContentEditable(target: ContentEditableTarget, replacement: string): AppliedEdit {
  const text = replacement.replace(/\r\n?/g, '\n');
  const doc = target.host.ownerDocument;
  selectRange(target);
  // One command = one undo step: plain text uses insertText, multi-line text
  // goes in as escaped HTML with <br> so it is still a single edit.
  const native = text.includes('\n')
    ? (() => {
        try {
          return doc.execCommand('insertHTML', false, escapeHtml(text).replace(/\n/g, '<br>'));
        } catch {
          return false;
        }
      })()
    : execInsertText(doc, text);

  if (native) {
    return {
      undo() {
        if (!target.host.isConnected) return false;
        target.host.focus({ preventScroll: true });
        try {
          return doc.execCommand('undo');
        } catch {
          return false;
        }
      },
    };
  }

  const inserted = insertDirectly(target, text);
  return {
    undo() {
      const first = inserted[0];
      const last = inserted[inserted.length - 1];
      if (!first || !last || !first.isConnected || !last.isConnected) return false;
      const range = doc.createRange();
      range.setStartBefore(first);
      range.setEndAfter(last);
      range.deleteContents();
      range.insertNode(textFragment(doc, target.text));
      target.host.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'historyUndo' }));
      return true;
    },
  };
}

/** Select the captured text again, as it was when captured (the caller has checked it is unchanged). */
export function reselect(target: EditTarget): void {
  if (target.kind === 'content-editable') {
    selectRange(target);
    return;
  }
  target.element.focus({ preventScroll: true });
  target.element.setSelectionRange(target.start, target.end);
}

/** Replace the captured selection with `replacement`, or throw TEXT_CHANGED. */
export function applyRewrite(target: EditTarget, replacement: string): AppliedEdit {
  if (!isUnchanged(target)) throw new EchoError('TEXT_CHANGED');
  return target.kind === 'text-control' ? applyToTextControl(target, replacement) : applyToContentEditable(target, replacement);
}
