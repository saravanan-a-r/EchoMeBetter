/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { ForegroundController, type JobPortLike, type SetupPortLike } from '../../../foreground/controller';
import { OverlayApp } from '../../../foreground/ui/OverlayApp';
import { OverlayStore, type SetupState } from '../../../foreground/ui/overlayStore';
import { setupCopy } from '../../../foreground/ui/SetupCard';
import type { JobEvent, SetupEvent, SetupRequest } from '../../../shared/messages';

function setup() {
  document.body.innerHTML = '<textarea></textarea>';
  const textarea = document.querySelector('textarea')!;
  textarea.value = 'pls send the deck tmrw';
  textarea.focus();
  textarea.setSelectionRange(0, textarea.value.length);

  let deliverJob!: (message: unknown) => void;
  const jobRequests: unknown[] = [];
  const jobPort: JobPortLike = {
    postMessage: (request) => void jobRequests.push(request),
    disconnect: jest.fn(),
    onMessage: { addListener: (listener) => (deliverJob = listener) },
    onDisconnect: { addListener: () => undefined },
  };
  const setupPorts: { port: SetupPortLike; sent: SetupRequest[]; emit: (event: SetupEvent) => void; drop: () => void; disconnect: jest.Mock }[] = [];
  const connectSetup = jest.fn(() => {
    let deliver!: (message: unknown) => void;
    let drop!: () => void;
    const sent: SetupRequest[] = [];
    const disconnect = jest.fn();
    let gone = false;
    const port: SetupPortLike = {
      postMessage: (request) => {
        // Chrome refuses a port that has disconnected.
        if (gone) throw new Error('Attempting to use a disconnected port object');
        sent.push(request);
      },
      disconnect,
      onMessage: { addListener: (listener) => (deliver = listener) },
      onDisconnect: { addListener: (listener) => (drop = listener) },
    };
    setupPorts.push({ port, sent, emit: (event) => deliver(event), drop: () => {
        gone = true;
        drop();
      }, disconnect });
    return port;
  });
  const store = new OverlayStore();
  const openSettings = jest.fn();
  const controller = new ForegroundController({
    doc: document,
    win: window,
    connect: () => jobPort,
    overlay: () => store,
    writeClipboard: async () => undefined,
    openSettings,
    requestRewrite: () => undefined,
    connectSetup,
  });
  return { store, controller, openSettings, connectSetup, setupPorts, jobRequests, emitJob: (event: JobEvent) => deliverJob(event), textarea };
}

const noModel = { kind: 'echo/notice', error: { code: 'MODEL_NOT_DOWNLOADED' }, style: 'concise' } as const;

describe('the in-place model download, from the controller', () => {
  test('a rewrite that needs the model opens the download card instead of an error toast, and follows it to the end', () => {
    const { store, controller, setupPorts, openSettings } = setup();
    controller.handleMessage(noModel);
    expect(store.getSnapshot().toast).toBeNull();
    expect(store.getSnapshot().setup).toMatchObject({ subject: { kind: 'model' }, style: 'concise', view: { phase: 'checking' } });
    const [card] = setupPorts;
    expect(card!.sent).toEqual([{ kind: 'setup/open', need: { kind: 'model', style: 'concise' } }]);

    card!.emit({ kind: 'setup/offer', totalBytes: 866e6, styles: ['concise', 'professional'] });
    expect(store.getSnapshot().setup!.view).toEqual({ phase: 'offer', totalBytes: 866e6, styles: ['concise', 'professional'] });
    store.getSnapshot().setup!.download();
    expect(card!.sent.at(-1)).toEqual({ kind: 'setup/download' });

    card!.emit({ kind: 'setup/progress', verifying: false, receivedBytes: 100, totalBytes: 1000, bytesPerSecond: 10 });
    expect(store.getSnapshot().setup!.view).toMatchObject({ phase: 'downloading', receivedBytes: 100 });
    card!.emit({ kind: 'setup/done' });
    expect(store.getSnapshot().setup!.view).toEqual({ phase: 'done' });

    store.getSnapshot().setup!.openSettings();
    expect(openSettings).toHaveBeenCalledWith('performance');
    expect(store.getSnapshot().setup).toBeNull();
    expect(card!.disconnect).toHaveBeenCalled();
  });

  test('once it is installed, "Rewrite now" starts the rewrite the user asked for, on the text still selected', () => {
    const { store, controller, setupPorts, jobRequests, textarea } = setup();
    controller.handleMessage(noModel);
    setupPorts[0]!.emit({ kind: 'setup/done' });
    expect(textarea.selectionStart).toBe(0);
    store.getSnapshot().setup!.rewrite();
    expect(store.getSnapshot().setup).toBeNull();
    expect(setupPorts[0]!.disconnect).toHaveBeenCalled();
    expect(jobRequests).toEqual([{ kind: 'job/request', jobId: expect.stringMatching(/^page-/), style: 'concise', text: 'pls send the deck tmrw' }]);
    expect(store.getSnapshot().working).toMatchObject({ style: 'concise' });
  });

  test('the style comes along when the worker is the one that finds the model missing', () => {
    const { store, controller, emitJob, setupPorts } = setup();
    controller.start('job-1', 'friendly');
    emitJob({ kind: 'job/failed', jobId: 'job-1', error: { code: 'MODEL_NOT_DOWNLOADED' } });
    expect(store.getSnapshot().setup).toMatchObject({ style: 'friendly' });
    expect(setupPorts[0]!.sent).toEqual([{ kind: 'setup/open', need: { kind: 'model', style: 'friendly' } }]);
  });

  test('without a known style there is nothing to carry on with', () => {
    const { store, controller } = setup();
    controller.handleMessage({ kind: 'echo/notice', error: { code: 'MODEL_NOT_DOWNLOADED' } });
    expect(store.getSnapshot().setup).toMatchObject({ style: null });
  });

  test('a missing style names the style it needs', () => {
    const { store, controller, setupPorts } = setup();
    controller.handleMessage({ kind: 'echo/notice', error: { code: 'STYLE_NOT_DOWNLOADED', details: { style: 'Grammar', adapter: 'Professional' } }, style: 'grammar' });
    expect(store.getSnapshot().setup).toMatchObject({ subject: { kind: 'style', name: 'Professional' }, style: 'grammar' });
    expect(setupPorts[0]!.sent).toEqual([{ kind: 'setup/open', need: { kind: 'style', style: 'professional' } }]);
  });

  test('the same happens when the worker is the one that finds the model missing', () => {
    const { store, controller, emitJob } = setup();
    controller.start('job-1', 'professional');
    emitJob({ kind: 'job/failed', jobId: 'job-1', error: { code: 'MODEL_NOT_DOWNLOADED' } });
    expect(store.getSnapshot().working).toBeNull();
    expect(store.getSnapshot().toast).toBeNull();
    expect(store.getSnapshot().setup).toMatchObject({ subject: { kind: 'model' } });
  });

  test('other errors, and a page without the service worker connection, still get the toast', () => {
    const { store, controller } = setup();
    controller.handleMessage({ kind: 'echo/notice', error: { code: 'FRAME_INACCESSIBLE' } });
    expect(store.getSnapshot().setup).toBeNull();
    expect(store.getSnapshot().toast).toMatchObject({ tone: 'error' });

    document.body.innerHTML = '';
    const bare = new OverlayStore();
    const without = new ForegroundController({ doc: document, win: window, connect: () => { throw new Error('no'); }, overlay: () => bare, writeClipboard: async () => undefined, openSettings: () => undefined, requestRewrite: () => undefined });
    without.handleMessage(noModel);
    expect(bare.getSnapshot().setup).toBeNull();
    expect(bare.getSnapshot().toast?.message).toContain('Download the writing model first');
  });

  test('a new rewrite closes the card and its connection; a service worker that went away is shown as a failure that can be retried', () => {
    const { store, controller, setupPorts } = setup();
    controller.handleMessage(noModel);
    controller.start('job-2', 'professional');
    expect(store.getSnapshot().setup).toBeNull();
    expect(setupPorts[0]!.disconnect).toHaveBeenCalled();

    controller.handleMessage(noModel);
    setupPorts[1]!.drop();
    expect(store.getSnapshot().setup!.view).toMatchObject({ phase: 'failed', error: { code: 'DISCONNECTED' } });
    store.getSnapshot().setup!.download();
    expect(setupPorts).toHaveLength(3);
  });
});

describe('the download card', () => {
  const state = (view: SetupState['view'], subject: SetupState['subject'] = { kind: 'model' }, style: SetupState['style'] = null) => {
    const calls = { download: jest.fn(), rewrite: jest.fn(), openSettings: jest.fn(), close: jest.fn() };
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    act(() => store.showSetup({ subject, style, anchor: { top: 100, left: 100, bottom: 130, right: 400 }, view, ...calls }));
    return { store, ...calls };
  };

  test('says why, with the size on the button', () => {
    const { download, close } = state({ phase: 'offer', totalBytes: 866e6 });
    expect(screen.getByRole('group', { name: 'Download the writing model' })).toBeInTheDocument();
    expect(screen.getByText(/needs its writing model first/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download · 866 MB' }));
    expect(download).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(close).toHaveBeenCalledTimes(1);
  });

  test('names the styles that come with the writing model', () => {
    state({ phase: 'offer', totalBytes: 939e6, styles: ['concise', 'professional'] });
    expect(screen.getByText(/This one-time download includes the Concise and Professional styles and stays on this device\./)).toBeInTheDocument();
  });

  test('shows progress with size, speed and time left, and says the download goes on if the card is closed', () => {
    state({ phase: 'downloading', verifying: false, receivedBytes: 217e6, totalBytes: 866e6, bytesPerSecond: 24e6 });
    expect(screen.getByRole('progressbar', { name: 'Download progress' })).toHaveAttribute('aria-valuenow', '25');
    expect(screen.getByText('217 MB of 866 MB · 24 MB/s · about 27 s left')).toBeInTheDocument();
    expect(screen.getByText(/The download continues in the background/)).toBeInTheDocument();
    expect(screen.getByText('Downloading the writing model · 25%')).toBeInTheDocument();
  });

  test('when the model is installed it says where the processor is chosen', () => {
    const { openSettings } = state({ phase: 'done' });
    expect(screen.getByText('Writing model installed')).toBeInTheDocument();
    expect(screen.getByText(/uses your GPU when one is available, and your CPU otherwise/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open settings' }));
    expect(openSettings).toHaveBeenCalledTimes(1);
  });

  test('when the user came to rewrite in a style, that is the main button, ahead of settings', () => {
    const { rewrite, openSettings } = state({ phase: 'done' }, { kind: 'model' }, 'concise');
    const buttons = screen.getAllByRole('button').map((button) => button.textContent);
    expect(buttons).toEqual(['Rewrite now · Concise', 'Open settings', '']);
    fireEvent.click(screen.getByRole('button', { name: 'Rewrite now · Concise' }));
    expect(rewrite).toHaveBeenCalledTimes(1);
    expect(openSettings).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Done' })).not.toBeInTheDocument();
  });

  test('an installed style offers the rewrite and a way to leave it', () => {
    const { close } = state({ phase: 'done' }, { kind: 'style', name: 'Professional' }, 'grammar');
    expect(screen.getByRole('button', { name: 'Rewrite now · Grammar' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open settings' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    expect(close).toHaveBeenCalledTimes(1);
  });

  test('a style that is installed needs no settings button', () => {
    state({ phase: 'done' }, { kind: 'style', name: 'Grammar' });
    expect(screen.getByText('Grammar style installed')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open settings' })).not.toBeInTheDocument();
  });

  test('a failure says what went wrong and can be retried', () => {
    const { download } = state({ phase: 'failed', error: { code: 'DOWNLOAD_FAILED' } });
    expect(screen.getByText("Download didn't finish")).toBeInTheDocument();
    expect(screen.getByText(/Check your connection, then try again/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(download).toHaveBeenCalledTimes(1);
  });

  test('the ready card closes itself after a while', () => {
    jest.useFakeTimers();
    try {
      const { close } = state({ phase: 'done' });
      act(() => void jest.advanceTimersByTime(30_100));
      expect(close).toHaveBeenCalledTimes(1);
    } finally {
      jest.useRealTimers();
    }
  });

  test('every moment has its words', () => {
    const base = { subject: { kind: 'model' } as const, style: null, anchor: null, download: () => undefined, rewrite: () => undefined, openSettings: () => undefined, close: () => undefined };
    expect(setupCopy({ ...base, view: { phase: 'downloading', verifying: true, receivedBytes: 1, totalBytes: 1, bytesPerSecond: 0 } }).title).toBe('Verifying the download');
    expect(setupCopy({ ...base, view: { phase: 'downloading', verifying: false, receivedBytes: 0, totalBytes: 0, bytesPerSecond: 0 } }).body).toBe('Connecting…');
  });
});
