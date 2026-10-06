/**
 * @jest-environment jsdom
 *
 * The page keeps every key press except a matching shortcut made with text
 * selected in an editable field.
 */
import { afterEach, beforeAll, describe, expect, jest, test } from '@jest/globals';
import { hasEditableSelection, listenForShortcuts } from '../../../foreground/shortcutListener';
import type { KeyPlatform } from '../../../shared/shortcuts';
import { installContentEditableSupport } from '../../helpers/dom';

beforeAll(installContentEditableSupport);

let stop: () => void = () => undefined;
afterEach(() => stop());

function setup(options: { platform?: KeyPlatform; enabled?: boolean } = {}) {
  const trigger = jest.fn();
  const flags = { enabled: options.enabled ?? true };
  stop = listenForShortcuts({ win: window, platform: options.platform ?? 'other', enabled: () => flags.enabled, trigger });
  // What the website itself would do with the same keys.
  const pageHandler = jest.fn();
  document.addEventListener('keydown', pageHandler);
  const press = (init: KeyboardEventInit, target: EventTarget = document.activeElement ?? document.body) => {
    const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
    target.dispatchEvent(event);
    document.removeEventListener('keydown', pageHandler);
    document.addEventListener('keydown', pageHandler);
    return event;
  };
  return { trigger, flags, pageHandler, press };
}

function textareaWithSelection(start = 0, end = 5) {
  document.body.innerHTML = '<textarea>hello there</textarea>';
  const textarea = document.querySelector('textarea')!;
  textarea.focus();
  textarea.setSelectionRange(start, end);
  return textarea;
}

const altShiftP = { key: 'P', code: 'KeyP', altKey: true, shiftKey: true };

describe('shortcut listener', () => {
  test('a shortcut on selected text starts that style and is kept from the page', () => {
    textareaWithSelection();
    const { trigger, pageHandler, press } = setup();
    const event = press(altShiftP);
    expect(trigger).toHaveBeenCalledWith('professional');
    expect(event.defaultPrevented).toBe(true);
    expect(pageHandler).not.toHaveBeenCalled();
  });

  test('without selected text the page gets the keys, untouched', () => {
    textareaWithSelection(3, 3);
    const { trigger, pageHandler, press } = setup();
    const event = press(altShiftP);
    expect(trigger).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
    expect(pageHandler).toHaveBeenCalledTimes(1);
  });

  test('turned off, nothing is touched', () => {
    textareaWithSelection();
    const { trigger, pageHandler, press } = setup({ enabled: false });
    const event = press(altShiftP);
    expect(trigger).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
    expect(pageHandler).toHaveBeenCalledTimes(1);
  });

  test('other keys always reach the page', () => {
    textareaWithSelection();
    const { trigger, pageHandler, press } = setup();
    press({ key: 'P', code: 'KeyP', ctrlKey: true, shiftKey: true });
    press({ key: 'b', code: 'KeyB' });
    expect(trigger).not.toHaveBeenCalled();
    expect(pageHandler).toHaveBeenCalledTimes(2);
  });

  test('macOS listens for Control+Shift', () => {
    textareaWithSelection();
    const { trigger, press } = setup({ platform: 'mac' });
    press({ key: 'E', code: 'KeyE', ctrlKey: true, shiftKey: true });
    press(altShiftP);
    expect(trigger.mock.calls).toEqual([['elaborate']]);
  });

  test('holding the keys down starts one rewrite', () => {
    textareaWithSelection();
    const { trigger, press } = setup();
    press(altShiftP);
    const repeated = press({ ...altShiftP, repeat: true });
    expect(trigger).toHaveBeenCalledTimes(1);
    expect(repeated.defaultPrevented).toBe(true);
  });

  test('selected text in a contenteditable editor counts', () => {
    document.body.innerHTML = '<div contenteditable="true"><p>hello there</p></div>';
    const editor = document.querySelector('div')!;
    editor.focus();
    const range = document.createRange();
    range.selectNodeContents(editor.querySelector('p')!);
    document.getSelection()!.removeAllRanges();
    document.getSelection()!.addRange(range);
    expect(hasEditableSelection(document)).toBe(true);
    const { trigger, press } = setup();
    press({ key: 'G', code: 'KeyG', altKey: true, shiftKey: true }, editor);
    expect(trigger).toHaveBeenCalledWith('grammar');
  });

  test('passwords, read-only fields and page text are left alone', () => {
    document.body.innerHTML = '<input type="password" value="secret123">';
    const password = document.querySelector('input')!;
    password.focus();
    password.setSelectionRange(0, 6);
    expect(hasEditableSelection(document)).toBe(false);

    const textarea = textareaWithSelection();
    textarea.readOnly = true;
    expect(hasEditableSelection(document)).toBe(false);

    document.body.innerHTML = '<p>plain page text</p>';
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('p')!);
    document.getSelection()!.removeAllRanges();
    document.getSelection()!.addRange(range);
    expect(hasEditableSelection(document)).toBe(false);
  });
});
