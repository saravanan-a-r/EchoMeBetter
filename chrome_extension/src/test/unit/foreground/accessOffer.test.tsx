/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ForegroundController, TOAST_MS, type JobPortLike } from '../../../foreground/controller';
import { OverlayApp } from '../../../foreground/ui/OverlayApp';
import { OverlayStore, type SiteAccessOffer } from '../../../foreground/ui/overlayStore';
import type { AccessOfferReply, AccessResult, JobEvent } from '../../../shared/messages';

const flush = () => act(async () => void (await Promise.resolve()));

/** The controller with a rewrite written and waiting for Replace; `reply` is what the service worker answers to a Replace. */
function written(reply: AccessOfferReply | Promise<AccessOfferReply>, result: AccessResult = 'granted') {
  document.body.innerHTML = '<textarea></textarea>';
  const textarea = document.querySelector('textarea')!;
  textarea.value = 'pls send the deck tmrw';
  textarea.focus();
  textarea.setSelectionRange(0, textarea.value.length);
  let deliver!: (message: unknown) => void;
  const port: JobPortLike = {
    postMessage: () => undefined,
    disconnect: jest.fn(),
    onMessage: { addListener: (listener) => (deliver = listener) },
    onDisconnect: { addListener: () => undefined },
  };
  const store = new OverlayStore();
  const afterReplace = jest.fn(async () => reply);
  const request = jest.fn(async () => result);
  const controller = new ForegroundController({
    doc: document,
    win: window,
    connect: () => port,
    overlay: () => store,
    writeClipboard: async () => undefined,
    openSettings: () => undefined,
    requestRewrite: () => undefined,
    siteAccess: { afterReplace, request },
  });
  const emit = (event: JobEvent) => deliver(event);
  controller.start('job-1', 'professional');
  emit({ kind: 'job/partial', jobId: 'job-1', text: 'Please send the deck tomorrow.' });
  emit({ kind: 'job/done', jobId: 'job-1', text: 'Please send the deck tomorrow.' });
  const replace = () => textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, composed: true }));
  return { store, afterReplace, request, replace, textarea };
}

const OFFER: AccessOfferReply = { offer: true, hold: true, shortcuts: true, platform: 'other' };

describe('the hold and shortcuts offer, from the controller', () => {
  test('Replace shows "Text replaced" at once; when the service worker says so, the toast grows into the offer and Allow asks for access', async () => {
    const { store, afterReplace, request, replace, textarea } = written(OFFER);
    replace();
    expect(textarea.value).toBe('Please send the deck tomorrow.');
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Text replaced', compact: true });
    expect(store.getSnapshot().toast?.offer).toBeUndefined();
    expect(afterReplace).toHaveBeenCalledTimes(1);

    await flush();
    const toast = store.getSnapshot().toast!;
    expect(toast).toMatchObject({ title: 'Text replaced', compact: true, durationMs: TOAST_MS.offer, offer: { hold: true, shortcuts: true, platform: 'other' } });
    expect(toast.actions.map((action) => action.label)).toEqual(['Undo']);
    await expect(toast.offer!.allow()).resolves.toBe('granted');
    expect(request).toHaveBeenCalledTimes(1);

    toast.actions[0]!.run();
    expect(textarea.value).toBe('pls send the deck tmrw');
  });

  test('no offer: the toast stays as it is', async () => {
    const { store, replace } = written({ offer: false });
    replace();
    await flush();
    expect(store.getSnapshot().toast?.offer).toBeUndefined();
  });

  test('an answer that arrives after the user pressed Undo (or closed the toast) is dropped', async () => {
    let answer!: (reply: AccessOfferReply) => void;
    const { store, replace } = written(new Promise<AccessOfferReply>((resolve) => (answer = resolve)));
    replace();
    store.getSnapshot().toast!.actions[0]!.run();
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Original text restored' });
    answer(OFFER);
    await flush();
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Original text restored' });
    expect(store.getSnapshot().toast?.offer).toBeUndefined();
  });

  test('a failing service worker never gets in the way of the toast', async () => {
    const { store, replace, afterReplace } = written(OFFER);
    afterReplace.mockRejectedValueOnce(new Error('Extension context invalidated.'));
    replace();
    await flush();
    expect(store.getSnapshot().toast).toMatchObject({ title: 'Text replaced' });
  });
});

describe('the offer card', () => {
  const show = (offer: Partial<SiteAccessOffer> = {}, allow: () => Promise<AccessResult> = async () => 'granted') => {
    const store = new OverlayStore();
    const undo = jest.fn();
    render(<OverlayApp store={store} />);
    act(() => {
      store.showToast({
        tone: 'success',
        title: 'Text replaced',
        compact: true,
        actions: [{ label: 'Undo', run: undo }],
        anchor: { top: 100, left: 100, bottom: 130, right: 400 },
        durationMs: 0,
        offer: { hold: true, shortcuts: true, platform: 'other', allow, ...offer },
      });
    });
    return { store, undo };
  };

  test('keeps "Text replaced" and Undo on its first line, and shows both features with their keys', () => {
    const { undo } = show();
    expect(screen.getByText('Text replaced')).toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Rewrite faster, without the right-click menu' })).toBeInTheDocument();
    expect(screen.getByText('Press and hold')).toBeInTheDocument();
    expect(screen.getByText('Select text, then press Alt+Shift+P for Professional.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(undo).toHaveBeenCalledTimes(1);
  });

  test('shows the Mac keys on a Mac, and only the feature that is switched on', () => {
    show({ platform: 'mac', hold: false });
    expect(screen.getByText('Select text, then press Ctrl+Shift+P for Professional.')).toBeInTheDocument();
    expect(screen.queryByText('Press and hold')).not.toBeInTheDocument();
    expect(screen.getByRole('group', { name: 'Rewrite with a keyboard shortcut' })).toBeInTheDocument();
  });

  test('Allow waits for Chrome, then says it is on and closes itself', async () => {
    jest.useFakeTimers();
    try {
      let answer!: (result: AccessResult) => void;
      const allow = jest.fn(() => new Promise<AccessResult>((resolve) => (answer = resolve)));
      const { store } = show({}, allow);
      fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
      expect(allow).toHaveBeenCalledTimes(1);
      expect(screen.getByRole('button', { name: 'Waiting for Chrome…' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Not now' })).toBeDisabled();

      await act(async () => answer('granted'));
      expect(screen.getByRole('group', { name: 'Hold and shortcuts are on' })).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: 'Allow on websites' })).not.toBeInTheDocument();
      act(() => void jest.advanceTimersByTime(5100));
      expect(store.getSnapshot().toast).toBeNull();
    } finally {
      jest.useRealTimers();
    }
  });

  test('a "no" in Chrome says where to turn it on later', async () => {
    show({}, async () => 'denied');
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    await waitFor(() => expect(screen.getByRole('group', { name: 'Not turned on' })).toBeInTheDocument());
    expect(screen.getByText(/any time from the EchoMeBetter popup, under Settings/)).toBeInTheDocument();
  });

  test('when the popup opened instead, the card closes; Not now closes it too', async () => {
    const opened = show({}, async () => 'opened');
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    await waitFor(() => expect(opened.store.getSnapshot().toast).toBeNull());

    const later = show();
    fireEvent.click(screen.getAllByRole('button', { name: 'Not now' }).at(-1)!);
    expect(later.store.getSnapshot().toast).toBeNull();
  });
});
