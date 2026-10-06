/**
 * @jest-environment jsdom
 *
 * The page-side journey end to end: start → loader → events → text replaced →
 * toast with Undo, plus every way it can go differently (busy, nothing
 * selected, Esc, the extension going away, text edited meanwhile).
 */
import { beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { ForegroundController, type JobPortLike } from '../../../foreground/controller';
import { isBusyCursorOn } from '../../../foreground/ui/busyCursor';
import { OverlayStore } from '../../../foreground/ui/overlayStore';
import type { JobEvent, JobRequest } from '../../../shared/messages';
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
  const clock = { time: 0 };
  const controller = new ForegroundController({ doc: document, win: window, connect: () => port, overlay: () => store, writeClipboard, now: () => clock.time });
  const emit = (event: JobEvent) => deliver(event);
  return { clock, textarea, sent, port, store, controller, emit, dropConnection: () => dropConnection(), writeClipboard };
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

  test('the toast says how long the rewrite took, not counting model loading', () => {
    const { clock, store, controller, emit } = setup();
    controller.start('job-1', 'concise');
    clock.time = 5000;
    emit({ kind: 'job/phase', jobId: 'job-1', phase: 'rewriting' });
    clock.time = 5420;
    emit({ kind: 'job/done', jobId: 'job-1', text: 'Please send the deck.' });
    expect(store.getSnapshot().toast!.message).toBe('Took 420 ms. Not quite right? Undo restores your original.');
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
