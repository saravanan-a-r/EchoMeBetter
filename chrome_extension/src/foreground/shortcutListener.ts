/**
 * Listens for the style shortcuts on a page (see shared/shortcuts.ts).
 *
 * The page keeps every key press except a matching chord made while text is
 * selected in an editable field; only then is the event stopped and the
 * rewrite asked for. With shortcuts turned off nothing is touched at all.
 */
import { matchShortcut, type KeyPlatform } from '../shared/shortcuts';
import type { StyleId } from '../shared/styles';
import { deepActiveElement, editingHost, selectionFor, TEXT_INPUT_TYPES } from './target/capture';

export interface ShortcutListenerDeps {
  readonly win: Window;
  readonly platform: KeyPlatform;
  readonly enabled: () => boolean;
  readonly trigger: (style: StyleId) => void;
}

function isTextControl(element: Element): element is HTMLInputElement | HTMLTextAreaElement {
  if (element.tagName === 'TEXTAREA') return true;
  return element.tagName === 'INPUT' && TEXT_INPUT_TYPES.has(((element as HTMLInputElement).getAttribute('type') ?? 'text').toLowerCase());
}

/** Whether a rewrite started now would have text to work on. */
export function hasEditableSelection(doc: Document): boolean {
  const active = deepActiveElement(doc);
  if (active && isTextControl(active)) {
    if (active.readOnly || active.disabled) return false;
    const { selectionStart: start, selectionEnd: end } = active;
    return start !== null && end !== null && active.value.slice(start, end).trim().length > 0;
  }
  const root = (active?.getRootNode() as Document | ShadowRoot | undefined) ?? doc;
  const selection = selectionFor(root, doc);
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return false;
  return editingHost(selection.getRangeAt(0).startContainer) !== null && selection.toString().trim().length > 0;
}

export function listenForShortcuts({ win, platform, enabled, trigger }: ShortcutListenerDeps): () => void {
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.isComposing || !enabled()) return;
    const style = matchShortcut(event, platform);
    if (!style || !hasEditableSelection(win.document)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    // Holding the keys down repeats the event; one rewrite is enough.
    if (!event.repeat) trigger(style);
  };
  // Capture phase on the window: ahead of the page's own handlers.
  win.addEventListener('keydown', onKeyDown, true);
  return () => win.removeEventListener('keydown', onKeyDown, true);
}
