/** @jest-environment jsdom */
import { beforeAll, beforeEach, describe, expect, test } from '@jest/globals';
import { captureTarget, splitWhitespace } from '../../../foreground/target/capture';
import { installContentEditableSupport, selectContents } from '../../helpers/dom';

beforeAll(installContentEditableSupport);
beforeEach(() => {
  document.body.innerHTML = '';
  document.getSelection()?.removeAllRanges();
});

function field<T extends HTMLInputElement | HTMLTextAreaElement>(html: string, start: number, end: number): T {
  document.body.innerHTML = html;
  const element = document.body.firstElementChild as T;
  element.focus();
  element.setSelectionRange(start, end);
  return element;
}

describe('captureTarget', () => {
  test('textarea: the selected range and its text', () => {
    const element = field('<textarea>hello brave world</textarea>', 6, 11);
    expect(captureTarget()).toEqual({ kind: 'text-control', element, start: 6, end: 11, text: 'brave' });
  });

  test.each(['text', 'search', 'url', 'tel'])('input[type=%s] is supported', (type) => {
    field(`<input type="${type}" value="hello there">`, 0, 5);
    expect(captureTarget()).toMatchObject({ kind: 'text-control', text: 'hello' });
  });

  test('password fields are refused without reading them', () => {
    document.body.innerHTML = '<input type="password" value="hunter2">';
    (document.body.firstElementChild as HTMLInputElement).focus();
    expect(() => captureTarget()).toThrow(expect.objectContaining({ code: 'PASSWORD_FIELD' }));
  });

  test('fields without a selection API (email, number) are refused', () => {
    document.body.innerHTML = '<input type="email" value="a@b.c">';
    (document.body.firstElementChild as HTMLInputElement).focus();
    expect(() => captureTarget()).toThrow(expect.objectContaining({ code: 'UNSUPPORTED_FIELD' }));
  });

  test('read-only fields are refused', () => {
    field('<textarea readonly>locked text</textarea>', 0, 6);
    expect(() => captureTarget()).toThrow(expect.objectContaining({ code: 'UNSUPPORTED_FIELD' }));
  });

  test('an empty or whitespace-only selection asks the user to select text', () => {
    field('<textarea>a   b</textarea>', 1, 4);
    expect(() => captureTarget()).toThrow(expect.objectContaining({ code: 'NO_SELECTION' }));
  });

  test('contenteditable: a live range inside the editing host', () => {
    document.body.innerHTML = '<div contenteditable="true"><p>Dear team,</p></div>';
    const host = document.body.firstElementChild as HTMLElement;
    host.focus();
    selectContents(host.querySelector('p')!);
    const target = captureTarget();
    expect(target).toMatchObject({ kind: 'content-editable', host, text: 'Dear team,', rangeText: 'Dear team,' });
  });

  test('a selection running out of the editor is refused', () => {
    document.body.innerHTML = '<div contenteditable="true">inside</div><p>outside</p>';
    const host = document.body.firstElementChild as HTMLElement;
    host.focus();
    const range = document.createRange();
    range.setStart(host.firstChild!, 0);
    range.setEnd(document.querySelector('p')!.firstChild!, 3);
    document.getSelection()!.removeAllRanges(); // focusing placed a caret
    document.getSelection()!.addRange(range);
    expect(() => captureTarget()).toThrow(expect.objectContaining({ code: 'UNSUPPORTED_FIELD' }));
  });

  test('an input inside an open shadow root is found', () => {
    const outer = document.createElement('div');
    document.body.appendChild(outer);
    const shadow = outer.attachShadow({ mode: 'open' });
    shadow.innerHTML = '<textarea>shadow text</textarea>';
    const textarea = shadow.querySelector('textarea')!;
    textarea.focus();
    textarea.setSelectionRange(0, 6);
    expect(captureTarget()).toMatchObject({ kind: 'text-control', element: textarea, text: 'shadow' });
  });
});

test('splitWhitespace keeps the page spacing out of the model input', () => {
  expect(splitWhitespace('  hello world \n')).toEqual({ lead: '  ', core: 'hello world', trail: ' \n' });
  expect(splitWhitespace('x')).toEqual({ lead: '', core: 'x', trail: '' });
});
