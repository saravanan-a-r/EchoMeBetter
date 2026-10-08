/**
 * @jest-environment jsdom
 *
 * A press held still on selected text opens the style menu; everything else
 * stays an ordinary click or drag that the page handles as usual.
 */
import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { HOLD_MS, HOLD_SLOP, listenForHold } from '../../../foreground/holdListener';

/** jsdom has no PointerEvent: a MouseEvent with the pointer fields the listener reads. */
function pointer(type: string, init: MouseEventInit & { pointerType?: string; pointerId?: number; isPrimary?: boolean } = {}): MouseEvent {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, composed: true, clientX: 2, clientY: 2, ...init });
  Object.defineProperties(event, {
    pointerType: { value: init.pointerType ?? 'mouse' },
    pointerId: { value: init.pointerId ?? 1 },
    isPrimary: { value: init.isPrimary ?? true },
  });
  return event;
}

let stop: () => void = () => undefined;
beforeEach(() => {
  jest.useFakeTimers();
});
afterEach(() => {
  stop();
  jest.useRealTimers();
});

function setup(options: { enabled?: boolean; select?: [number, number] } = {}) {
  document.body.innerHTML = '<textarea>please send the deck tomorrow</textarea>';
  const textarea = document.querySelector('textarea')!;
  textarea.focus();
  textarea.setSelectionRange(...(options.select ?? [0, 6]));
  const open = jest.fn();
  const flags = { enabled: options.enabled ?? true };
  stop = listenForHold({ win: window, enabled: () => flags.enabled, open });
  const pageSaw: string[] = [];
  for (const type of ['pointerdown', 'pointerup', 'mouseup']) textarea.addEventListener(type, () => pageSaw.push(type));
  return { textarea, open, flags, pageSaw };
}

describe('press and hold', () => {
  test('held still on the selection, it opens the menu there, for that selection; the page sees the press as usual', () => {
    const { textarea, open, pageSaw } = setup();
    textarea.dispatchEvent(pointer('pointerdown'));
    jest.advanceTimersByTime(HOLD_MS - 1);
    expect(open).not.toHaveBeenCalled();
    jest.advanceTimersByTime(1);
    expect(open).toHaveBeenCalledTimes(1);
    expect(open).toHaveBeenCalledWith({ x: 2, y: 2 }, expect.objectContaining({ kind: 'text-control', element: textarea, start: 0, end: 6, text: 'please' }));
    expect(pageSaw).toEqual(['pointerdown']);
  });

  test('the release that ends a hold keeps the selection: its default is prevented, though the page still hears it', () => {
    const { textarea, pageSaw } = setup();
    textarea.dispatchEvent(pointer('pointerdown'));
    jest.advanceTimersByTime(HOLD_MS);
    const mouseup = pointer('mouseup');
    textarea.dispatchEvent(mouseup);
    textarea.dispatchEvent(pointer('pointerup'));
    expect(mouseup.defaultPrevented).toBe(true);
    expect(pageSaw).toEqual(['pointerdown', 'mouseup', 'pointerup']);

    // Only that one release: the next click is the page's own.
    jest.advanceTimersByTime(1);
    const later = pointer('mouseup');
    textarea.dispatchEvent(later);
    expect(later.defaultPrevented).toBe(false);
  });

  test('letting go early is a click, and its release is left alone', () => {
    const { textarea, open } = setup();
    textarea.dispatchEvent(pointer('pointerdown'));
    jest.advanceTimersByTime(HOLD_MS / 2);
    const mouseup = pointer('mouseup');
    textarea.dispatchEvent(mouseup);
    textarea.dispatchEvent(pointer('pointerup'));
    jest.advanceTimersByTime(HOLD_MS);
    expect(open).not.toHaveBeenCalled();
    expect(mouseup.defaultPrevented).toBe(false);
  });

  test('moving is a drag, not a hold; a tremble within a few pixels still counts', () => {
    const drag = setup();
    drag.textarea.dispatchEvent(pointer('pointerdown'));
    window.dispatchEvent(pointer('pointermove', { clientX: 2 + HOLD_SLOP + 1, clientY: 2 }));
    jest.advanceTimersByTime(HOLD_MS);
    expect(drag.open).not.toHaveBeenCalled();
    stop();

    const tremble = setup();
    tremble.textarea.dispatchEvent(pointer('pointerdown'));
    window.dispatchEvent(pointer('pointermove', { clientX: 4, clientY: 3 }));
    jest.advanceTimersByTime(HOLD_MS);
    expect(tremble.open).toHaveBeenCalledTimes(1);
  });

  test.each([
    ['the right button', { button: 2 }],
    ['a modifier key (Shift extends the selection)', { shiftKey: true }],
    ['a finger: touch has its own long press', { pointerType: 'touch' }],
  ])('%s does not hold', (_label, init) => {
    const { textarea, open } = setup();
    textarea.dispatchEvent(pointer('pointerdown', init));
    jest.advanceTimersByTime(HOLD_MS);
    expect(open).not.toHaveBeenCalled();
  });

  test('without selected text, while turned off, or away from the selection, nothing opens', () => {
    const empty = setup({ select: [3, 3] });
    empty.textarea.dispatchEvent(pointer('pointerdown'));
    jest.advanceTimersByTime(HOLD_MS);
    expect(empty.open).not.toHaveBeenCalled();
    stop();

    const off = setup({ enabled: false });
    off.textarea.dispatchEvent(pointer('pointerdown'));
    jest.advanceTimersByTime(HOLD_MS);
    expect(off.open).not.toHaveBeenCalled();
    stop();

    // jsdom lays nothing out: the selection measures at the origin, so far from it is off the selection.
    const away = setup();
    away.textarea.dispatchEvent(pointer('pointerdown', { clientX: 400, clientY: 300 }));
    jest.advanceTimersByTime(HOLD_MS);
    expect(away.open).not.toHaveBeenCalled();
  });

  test('a selection gone by the time the hold completes opens nothing', () => {
    const { textarea, open } = setup();
    textarea.dispatchEvent(pointer('pointerdown'));
    textarea.setSelectionRange(2, 2);
    jest.advanceTimersByTime(HOLD_MS);
    expect(open).not.toHaveBeenCalled();
  });
});
