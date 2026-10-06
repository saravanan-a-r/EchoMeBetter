/** @jest-environment jsdom */
import { beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { applyRewrite } from '../../../foreground/target/apply';
import { captureTarget } from '../../../foreground/target/capture';
import { installContentEditableSupport, selectContents } from '../../helpers/dom';

beforeAll(installContentEditableSupport);
beforeEach(() => {
  document.body.innerHTML = '';
});

function textarea(value: string, start: number, end: number): HTMLTextAreaElement {
  document.body.innerHTML = `<textarea></textarea>`;
  const element = document.querySelector('textarea')!;
  element.value = value;
  element.focus();
  element.setSelectionRange(start, end);
  return element;
}

describe('text controls', () => {
  test('replaces exactly the selected range and notifies frameworks with an input event', () => {
    const element = textarea('Hi team, pls send it. Thanks', 9, 21);
    const onInput = jest.fn();
    element.addEventListener('input', onInput);
    applyRewrite(captureTarget(), 'Please send it.');
    expect(element.value).toBe('Hi team, Please send it. Thanks');
    expect(onInput).toHaveBeenCalled();
  });

  test('undo restores the original while the rewrite is untouched', () => {
    const element = textarea('abc old def', 4, 7);
    const edit = applyRewrite(captureTarget(), 'brand new');
    expect(element.value).toBe('abc brand new def');
    expect(edit.undo()).toBe(true);
    expect(element.value).toBe('abc old def');
  });

  test('undo declines once the user has edited the rewritten text', () => {
    const element = textarea('abc old def', 4, 7);
    const edit = applyRewrite(captureTarget(), 'new');
    element.value = 'abc NEW def';
    expect(edit.undo()).toBe(false);
    expect(element.value).toBe('abc NEW def');
  });

  test('nothing is written when the selected text changed meanwhile', () => {
    const element = textarea('keep this', 0, 4);
    const target = captureTarget();
    element.value = 'edit this';
    expect(() => applyRewrite(target, 'replaced')).toThrow(expect.objectContaining({ code: 'TEXT_CHANGED' }));
    expect(element.value).toBe('edit this');
  });

  test('a single-line input gets line breaks as spaces instead of losing them', () => {
    document.body.innerHTML = '<input type="text" value="subject line">';
    const input = document.querySelector('input')!;
    input.focus();
    input.setSelectionRange(0, 12);
    applyRewrite(captureTarget(), 'First line\nsecond line');
    expect(input.value).toBe('First line second line');
  });
});

describe('contenteditable', () => {
  test('replaces the selection, keeps line breaks as <br>, and undo restores it', () => {
    document.body.innerHTML = '<div contenteditable="true"><span>old words</span> stay</div>';
    const host = document.body.firstElementChild as HTMLElement;
    host.focus();
    selectContents(host.querySelector('span')!);
    const onInput = jest.fn();
    host.addEventListener('input', onInput);

    const edit = applyRewrite(captureTarget(), 'new\nlines');
    expect(host.innerHTML).toBe('<span>new<br>lines</span> stay');
    expect(onInput).toHaveBeenCalled();

    expect(edit.undo()).toBe(true);
    expect(host.textContent).toBe('old words stay');
  });

  test('a changed editor is left alone', () => {
    document.body.innerHTML = '<div contenteditable="true"><span>old</span></div>';
    const host = document.body.firstElementChild as HTMLElement;
    host.focus();
    selectContents(host.querySelector('span')!);
    const target = captureTarget();
    host.querySelector('span')!.textContent = 'changed';
    expect(() => applyRewrite(target, 'new')).toThrow(expect.objectContaining({ code: 'TEXT_CHANGED' }));
  });
});
