/**
 * @jest-environment jsdom
 *
 * The page-side journey end to end: start → loader → events → text replaced →
 * toast with Undo (with the live preview: the finished rewrite waits for
 * Replace or Discard first), plus every way it can go differently (busy,
 * nothing selected, Esc, the extension going away, text edited meanwhile).
 */
import { beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { ForegroundController, TOAST_MS, type JobPortLike } from '../../../foreground/controller';
import { leaveHeldSelection } from '../../../foreground/heldSelection';
import { captureTarget } from '../../../foreground/target/capture';
import { isBusyCursorOn } from '../../../foreground/ui/busyCursor';
import { OverlayStore } from '../../../foreground/ui/overlayStore';
import type { JobEvent, JobRequest, WokeFromRest } from '../../../shared/messages';
import { installContentEditableSupport } from '../../helpers/dom';

beforeAll(installContentEditableSupport);

function setup() {
  document.body.innerHTML = '<textarea></textarea>';
  const textarea = document.querySelector('textarea')!;
  textarea.value = 'Hi,  pls send the deck tmrw  thx';
  textarea.focus();
  textarea.setSelectionRange(4, 28); // "  pls send the deck tmrw" -- leading spaces included on purpose

  const sent: JobRequest[] = [];
  let deliver!: (message: unknown) => void;
  let dropConnection!: () => void;
  const port: JobPortLike = {
    postMessage: (message) => sent.push(message),
    disconnect: jest.fn(),
    onMessage: { addListener: (listener) => (deliver = listener) },
    onDisconnect: { addListener: (listener) => (dropConnection = listener) },
  };
  const store = new OverlayStore();
  const writeClipboard = jest.fn(async (_text: string) => undefined);
  const openSettings = jest.fn();
  const requestRewrite = jest.fn();
  const clock = { time: 0 };
  const controller = new ForegroundController({ doc: document, win: window, connect: () => port, overlay: () => store, writeClipboard, openSettings, requestRewrite, now: () => clock.time });
  const emit = (event: JobEvent) => deliver(event);
  return { clock, textarea, sent, port, store, controller, emit, dropConnection: () => dropConnection(), writeClipboard, openSettings, requestRewrite };
}

beforeEach(() => {
  document.documentElement.removeAttribute('data-echomebetter-busy');
});

describe('ForegroundController', () => {
  test('happy path: loader while working, rewrite applied with spacing kept, Undo restores', () => {
    const { textarea, sent, store, controller, emit, port } = setup();

    expect(controller.start('job-1', 'professional')).toEqual({ ok: true });
    expect(sent).toEqual([{ kind: 'job/request', jobId: 'job-1', style: 'professional', text: 'pls send the deck tmrw' }]);
    expect(store.getSnapshot().working).toMatchObject({ style: 'professional', phase: 'starting' });
    expect(isBusyCursorOn(document)).toBe(true);

    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'loading-model', progress: 0.4 });
    expect(store.getSnapshot().working).toMatchObject({ phase: 'loading-model', progress: 0.4 });

    emit({ kind: 'job/done', jobId: 'job-1', text: 'Please send the deck tomorrow.' });
    expect(textarea.value).toBe('Hi,  Please send the deck tomorrow.  thx');
    expect(store.getSnapshot().working).toBeNull();
    expect(isBusyCursorOn(document)).toBe(false);
    expect(port.disconnect).toHaveBeenCalled();

    const toast = store.getSnapshot().toast!;
    expect(toast).toMatchObject({ tone: 'success', title: 'Rewritten · Professional' });
    toast.actions.find((action) => action.label === 'Undo')!.run();
    expect(textarea.value).toBe('Hi,  pls send the deck tmrw  thx');
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Original text restored' });
  });

  test('the rewrite so far is only previewed; once written it waits for Replace, which changes the field once, to the final text', () => {
    const { clock, textarea, store, controller, emit, port } = setup();
    const onInput = jest.fn();
    textarea.addEventListener('input', onInput);
    controller.start('job-1', 'professional');
    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'rewriting' });
    emit({ kind: 'job/partial', jobId: 'job-1', text: 'Please send' });
    expect(store.getSnapshot().working).toMatchObject({ phase: 'rewriting', preview: 'Please send' });
    expect(textarea.value).toBe('Hi,  pls send the deck tmrw  thx');

    // The GPU failed the first attempt: the model loads again and the rewrite restarts.
    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'loading-model', progress: 0 });
    expect(store.getSnapshot().working?.preview).toBeUndefined();
    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'rewriting' });
    emit({ kind: 'job/partial', jobId: 'job-1', text: 'Please' });
    emit({ kind: 'job/partial', jobId: 'job-2', text: 'someone else' });
    expect(store.getSnapshot().working?.preview).toBe('Please');

    clock.time = 1200;
    emit({ kind: 'job/done', jobId: 'job-1', text: 'Please send the deck tomorrow.' });
    // Written: the job is over, the field untouched, the rewrite on screen to decide on.
    expect(store.getSnapshot().working).toBeNull();
    expect(store.getSnapshot().toast).toBeNull();
    expect(store.getSnapshot().review).toMatchObject({ style: 'professional', text: 'Please send the deck tomorrow.', detail: 'Took 1.2 s' });
    expect(controller.busy).toBe(false);
    expect(isBusyCursorOn(document)).toBe(false);
    expect(port.disconnect).toHaveBeenCalled();
    expect(onInput).not.toHaveBeenCalled();
    // Late text for a finished job is dropped.
    emit({ kind: 'job/partial', jobId: 'job-1', text: 'Please send the' });
    expect(store.getSnapshot().working).toBeNull();

    const enter = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
    textarea.dispatchEvent(enter);
    expect(enter.defaultPrevented).toBe(true);
    expect(textarea.value).toBe('Hi,  Please send the deck tomorrow.  thx');
    expect(onInput).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().review).toBeNull();
    const toast = store.getSnapshot().toast!;
    expect(toast).toMatchObject({ tone: 'success', title: 'Text replaced', durationMs: TOAST_MS.success });
    expect(toast.message).toBeUndefined();
    toast.actions.find((action) => action.label === 'Undo')!.run();
    expect(textarea.value).toBe('Hi,  pls send the deck tmrw  thx');
  });

  describe('a written rewrite waiting for a decision', () => {
    const written = (rest?: WokeFromRest) => {
      const parts = setup();
      parts.controller.start('job-1', 'friendly');
      parts.emit({ kind: 'job/phase', jobId: 'job-1', phase: 'rewriting' });
      parts.emit({ kind: 'job/partial', jobId: 'job-1', text: 'Could you' });
      parts.emit({ kind: 'job/done', jobId: 'job-1', text: 'Could you send the deck tomorrow?', wokeFromRest: rest });
      return parts;
    };
    const press = (target: EventTarget, key: string) => {
      const event = new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, composed: true });
      target.dispatchEvent(event);
      return event;
    };

    test('Esc or Discard leaves the text as it was, quietly', () => {
      const { textarea, store } = written();
      expect(press(textarea, 'Escape').defaultPrevented).toBe(true);
      expect(store.getSnapshot()).toMatchObject({ review: null, toast: null });
      expect(textarea.value).toBe('Hi,  pls send the deck tmrw  thx');
      // Its keys are the page's again.
      expect(press(textarea, 'Enter').defaultPrevented).toBe(false);

      const again = written();
      again.store.getSnapshot().review!.discard();
      expect(again.store.getSnapshot()).toMatchObject({ review: null, toast: null });
      expect(again.textarea.value).toBe('Hi,  pls send the deck tmrw  thx');
    });

    test('Enter and Esc stay the page’s outside the edited field, and Enter once the text there has changed', () => {
      const { textarea, store } = written();
      document.body.insertAdjacentHTML('beforeend', '<input id="elsewhere">');
      const elsewhere = document.querySelector<HTMLInputElement>('#elsewhere')!;
      expect(press(elsewhere, 'Enter').defaultPrevented).toBe(false);
      expect(press(elsewhere, 'Escape').defaultPrevented).toBe(false);
      expect(press(textarea, 'Enter').defaultPrevented).toBe(true);
      expect(store.getSnapshot().toast).toMatchObject({ title: 'Text replaced' });

      const typedOver = written();
      typedOver.textarea.value = 'Hi, typed something else';
      expect(press(typedOver.textarea, 'Enter').defaultPrevented).toBe(false);
      expect(typedOver.store.getSnapshot().review).not.toBeNull();
    });

    test('Replace after the text changed leaves it alone and offers the rewrite to copy', () => {
      const { textarea, store, writeClipboard } = written();
      textarea.value = 'Something else entirely';
      store.getSnapshot().review!.replace();
      expect(textarea.value).toBe('Something else entirely');
      const toast = store.getSnapshot().toast!;
      expect(toast.message).toMatch(/changed while it was being rewritten/);
      toast.actions.find((action) => action.label === 'Copy rewrite')!.run();
      expect(writeClipboard).toHaveBeenCalledWith('Could you send the deck tomorrow?');
    });

    test('the wake-up note comes with the rewrite, and its button leaves the decision open', () => {
      const { store, openSettings } = written({ idleMinutes: 15 });
      const review = store.getSnapshot().review!;
      expect(review.note?.text).toMatch(/^Waking up took .+ EchoMeBetter rests after 15 minutes without use/);
      review.note!.action.run();
      expect(openSettings).toHaveBeenCalledWith('keep-awake');
      expect(store.getSnapshot().review).toBe(review);
      review.replace();
      expect(store.getSnapshot().toast?.note).toBeUndefined();
    });

    test('a new rewrite or the style menu sets it aside, and its keys go with it', () => {
      const { store, controller, sent } = written();
      expect(controller.start('job-2', 'concise')).toEqual({ ok: true });
      expect(store.getSnapshot()).toMatchObject({ review: null, working: { style: 'concise' } });
      expect(sent.at(-1)).toMatchObject({ kind: 'job/request', jobId: 'job-2' });

      const other = written();
      other.controller.openMenu({ x: 1, y: 1 });
      expect(other.store.getSnapshot().review).toBeNull();
      other.controller.closeMenu();
      expect(press(other.textarea, 'Enter').defaultPrevented).toBe(false);
      expect(other.textarea.value).toBe('Hi,  pls send the deck tmrw  thx');
    });
  });

  test('the toast says how long the rewrite took, not counting model loading', () => {
    const { clock, store, controller, emit } = setup();
    controller.start('job-1', 'concise');
    clock.time = 5000;
    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'rewriting' });
    clock.time = 5420;
    emit({ kind: 'job/done', jobId: 'job-1', text: 'Please send the deck.' });
    expect(store.getSnapshot().toast!.message).toBe('Took 420 ms. Not quite right? Undo restores your original.');
  });

  test('a rewrite that had to wake the model from its rest says how long that took, and offers to keep it awake longer', () => {
    const { clock, store, controller, emit, openSettings } = setup();
    controller.start('job-1', 'concise');
    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'loading-model', progress: 0 });
    clock.time = 3400;
    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'rewriting' });
    clock.time = 3820;
    emit({ kind: 'job/done', jobId: 'job-1', text: 'Please send the deck.', wokeFromRest: { idleMinutes: 15 } });

    const toast = store.getSnapshot().toast!;
    expect(toast.message).toBe('Took 420 ms. Not quite right? Undo restores your original.');
    expect(toast.actions.map((action) => action.label)).toEqual(['Undo']);
    expect(toast.note?.text).toBe('Waking up took 3.4 s. EchoMeBetter rests after 15 minutes without use, to keep your computer fast.');
    expect(toast.durationMs).toBe(TOAST_MS.withNote);
    toast.note!.action.run();
    expect(openSettings).toHaveBeenCalledWith('keep-awake');
  });

  test('the note also comes with "looks good already", and not at all when the model was awake', () => {
    const first = setup();
    first.controller.start('job-1', 'grammar');
    first.emit({ kind: 'job/phase', jobId: 'job-1', phase: 'rewriting' });
    first.emit({ kind: 'job/done', jobId: 'job-1', text: 'pls send the deck tmrw', wokeFromRest: { idleMinutes: 60 } });
    expect(first.store.getSnapshot().toast).toMatchObject({ title: 'Looks good already', note: { text: expect.stringContaining('after 1 hour without use') } });

    const second = setup();
    second.controller.start('job-2', 'concise');
    second.emit({ kind: 'job/phase', jobId: 'job-2', phase: 'rewriting' });
    second.emit({ kind: 'job/done', jobId: 'job-2', text: 'Please send the deck.' });
    expect(second.store.getSnapshot().toast!.note).toBeUndefined();
    expect(second.store.getSnapshot().toast!.durationMs).toBe(TOAST_MS.success);
  });

  test('events for another job are ignored', () => {
    const { textarea, controller, emit } = setup();
    controller.start('job-1', 'concise');
    emit({ kind: 'job/done', jobId: 'someone-else', text: 'nope' });
    expect(textarea.value).toContain('pls send');
    expect(controller.busy).toBe(true);
  });

  test('a second request while working is refused with a gentle note', () => {
    const { controller, store } = setup();
    controller.start('job-1', 'concise');
    expect(controller.start('job-2', 'friendly')).toEqual({ ok: false, error: { code: 'BUSY', details: {} } });
    expect(store.getSnapshot().toast).toMatchObject({ tone: 'info', title: 'One at a time' });
  });

  test('nothing selected: no job is sent and the user is told what to do', () => {
    const { textarea, controller, sent, store } = setup();
    textarea.setSelectionRange(3, 3);
    expect(controller.start('job-1', 'concise')).toMatchObject({ ok: false, error: { code: 'NO_SELECTION' } });
    expect(sent).toEqual([]);
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Nothing selected' });
  });

  test('Esc cancels: the engine is told, the page is untouched, the loader goes away', () => {
    const { textarea, controller, sent, store } = setup();
    controller.start('job-1', 'elaborate');
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(sent.at(-1)).toEqual({ kind: 'job/cancel', jobId: 'job-1' });
    expect(controller.busy).toBe(false);
    expect(isBusyCursorOn(document)).toBe(false);
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Rewrite cancelled' });
    expect(textarea.value).toBe('Hi,  pls send the deck tmrw  thx');
  });

  test('the extension disappearing mid-job ends the job with an explanation', () => {
    const { controller, store, dropConnection } = setup();
    controller.start('job-1', 'grammar');
    dropConnection();
    expect(controller.busy).toBe(false);
    expect(store.getSnapshot().toast).toMatchObject({ tone: 'error', message: expect.stringContaining('updated or restarted') });
  });

  test('if the text was edited meanwhile, it is left alone and the rewrite can be copied', () => {
    const { textarea, controller, emit, store, writeClipboard } = setup();
    controller.start('job-1', 'grammar');
    textarea.value = 'Something else entirely, typed while waiting';
    emit({ kind: 'job/done', jobId: 'job-1', text: 'Please send the deck tomorrow.' });
    expect(textarea.value).toBe('Something else entirely, typed while waiting');
    const toast = store.getSnapshot().toast!;
    expect(toast.message).toMatch(/changed while it was being rewritten/);
    toast.actions.find((action) => action.label === 'Copy rewrite')!.run();
    expect(writeClipboard).toHaveBeenCalledWith('Please send the deck tomorrow.');
  });

  test('an unchanged result is reported instead of rewriting the field', () => {
    const { textarea, controller, emit, store } = setup();
    const onInput = jest.fn();
    textarea.addEventListener('input', onInput);
    controller.start('job-1', 'grammar');
    emit({ kind: 'job/done', jobId: 'job-1', text: 'pls send the deck tmrw' });
    expect(onInput).not.toHaveBeenCalled();
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Looks good already' });
  });

  test('engine failures are shown; a cancelled job stays quiet', () => {
    const { controller, emit, store } = setup();
    controller.start('job-1', 'grammar');
    emit({ kind: 'job/failed', jobId: 'job-1', error: { code: 'INPUT_TOO_LONG', details: { actual: 700, limit: 512 } } });
    expect(store.getSnapshot().toast).toMatchObject({ tone: 'error', message: expect.stringContaining('700 of 512') });

    store.dismissToast();
    controller.start('job-2', 'grammar');
    emit({ kind: 'job/failed', jobId: 'job-2', error: { code: 'CANCELLED' } });
    expect(store.getSnapshot().toast).toBeNull();
  });

  test('a notice from the service worker is shown as a toast', () => {
    const { controller, store } = setup();
    expect(controller.handleMessage({ kind: 'echo/notice', error: { code: 'FRAME_INACCESSIBLE' } })).toEqual({ ok: true });
    expect(store.getSnapshot().toast).toMatchObject({ message: expect.stringContaining('frame from another site') });
  });
});

describe('the style menu (press and hold)', () => {
  const POINT = { x: 40, y: 30 };
  const key = (init: KeyboardEventInit) => {
    const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
    document.activeElement!.dispatchEvent(event);
    return event;
  };

  test('opens with every style; Enter picks the first, which is asked for with the held text selected again', () => {
    const { textarea, controller, store, requestRewrite } = setup();
    expect(controller.handleMessage({ kind: 'echo/style-menu', point: POINT })).toEqual({ ok: true });
    const menu = store.getSnapshot().menu!;
    expect(menu.point).toEqual(POINT);
    expect(menu.styles).toEqual(['professional', 'grammar', 'friendly', 'concise', 'elaborate']);

    textarea.setSelectionRange(9, 9); // what letting go of the hold can do
    const enter = key({ key: 'Enter' });
    expect(enter.defaultPrevented).toBe(true);
    expect(requestRewrite).toHaveBeenCalledWith('professional');
    expect(store.getSnapshot().menu).toBeNull();
    expect([textarea.selectionStart, textarea.selectionEnd]).toEqual([4, 28]);
  });

  test('arrows move through the styles and wrap; a number picks directly; neither reaches the editor', () => {
    const { controller, store, requestRewrite } = setup();
    const pageKeys = jest.fn();
    document.addEventListener('keydown', pageKeys);
    controller.openMenu(POINT);
    key({ key: 'ArrowDown' });
    key({ key: 'ArrowDown' });
    expect(store.getSnapshot().menu!.active).toBe(2);
    key({ key: 'ArrowUp' });
    key({ key: 'ArrowUp' });
    key({ key: 'ArrowUp' });
    expect(store.getSnapshot().menu!.active).toBe(4);
    key({ key: '4' });
    expect(requestRewrite).toHaveBeenCalledWith('concise');
    expect(pageKeys).not.toHaveBeenCalled();
    document.removeEventListener('keydown', pageKeys);
  });

  test('a click on a style picks it', () => {
    const { controller, store, requestRewrite } = setup();
    controller.openMenu(POINT);
    store.getSnapshot().menu!.pick('friendly');
    expect(requestRewrite).toHaveBeenCalledWith('friendly');
    expect(controller.menuOpen).toBe(false);
  });

  test('Esc closes it without the page hearing the key; other keys are left to the page and the menu stays', () => {
    const { controller, store, requestRewrite } = setup();
    controller.openMenu(POINT);
    expect(key({ key: 'a' }).defaultPrevented).toBe(false);
    expect(controller.menuOpen).toBe(true);
    expect(key({ key: 'Escape' }).defaultPrevented).toBe(true);
    expect(store.getSnapshot().menu).toBeNull();
    expect(requestRewrite).not.toHaveBeenCalled();
  });

  test('a press anywhere outside it closes it; a press on the overlay itself does not', () => {
    const { controller } = setup();
    document.body.insertAdjacentHTML('beforeend', '<echomebetter-overlay><button>Friendly</button></echomebetter-overlay>');
    controller.openMenu(POINT);
    document.querySelector('echomebetter-overlay button')!.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, composed: true }));
    expect(controller.menuOpen).toBe(true);
    document.body.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, composed: true }));
    expect(controller.menuOpen).toBe(false);
  });

  test('it acts on the selection the hold was made on, even if the page dropped it meanwhile', () => {
    const { textarea, controller, requestRewrite } = setup();
    leaveHeldSelection(window, POINT, captureTarget(document));
    textarea.setSelectionRange(0, 0);
    controller.openMenu(POINT);
    expect([textarea.selectionStart, textarea.selectionEnd]).toEqual([4, 28]);
    key({ key: '2' });
    expect(requestRewrite).toHaveBeenCalledWith('grammar');
  });

  test('text typed over while it was open is not rewritten', () => {
    const { textarea, controller, requestRewrite } = setup();
    controller.openMenu(POINT);
    textarea.value = 'Hi, something new';
    key({ key: 'Enter' });
    expect(requestRewrite).not.toHaveBeenCalled();
    expect(controller.menuOpen).toBe(false);
  });

  test('during a rewrite it does not open: one at a time', () => {
    const { controller, store } = setup();
    controller.start('job-1', 'concise');
    controller.openMenu(POINT);
    expect(store.getSnapshot().menu).toBeNull();
    expect(store.getSnapshot().toast).toMatchObject({ title: 'One at a time' });
  });

  test('a rewrite started another way (a shortcut) closes it', () => {
    const { controller, store } = setup();
    controller.openMenu(POINT);
    controller.start('job-1', 'concise');
    expect(controller.menuOpen).toBe(false);
    expect(store.getSnapshot().menu).toBeNull();
  });
});
