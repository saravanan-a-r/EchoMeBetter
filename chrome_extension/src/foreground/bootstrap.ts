/**
 * Wires the foreground controller to Chrome. Idempotent: the service worker
 * injects this script on every menu click, and only the first injection in
 * a frame installs anything.
 */
import { isForegroundMessage, JOB_PORT_NAME } from '../shared/messages';
import { ForegroundController, type JobPortLike } from './controller';
import { mountOverlay, type Overlay } from './ui/mountOverlay';

const INSTALLED = Symbol.for('echomebetter.foreground');

function runtimeAlive(): boolean {
  try {
    return typeof chrome !== 'undefined' && Boolean(chrome.runtime?.id);
  } catch {
    return false;
  }
}

export function installForeground(win: Window = window): ForegroundController {
  const registry = win as unknown as Record<symbol, ForegroundController | undefined>;
  const existing = registry[INSTALLED];
  if (existing && runtimeAlive()) return existing;

  let overlay: Overlay | null = null;
  const controller = new ForegroundController({
    doc: win.document,
    win,
    connect: () => chrome.runtime.connect({ name: JOB_PORT_NAME }) as unknown as JobPortLike,
    overlay: () => {
      // Re-mount if the page removed our host element (some apps own all of <html>).
      if (!overlay || !overlay.host.isConnected) overlay = mountOverlay(win.document);
      return overlay.store;
    },
    writeClipboard: (text) => win.navigator.clipboard.writeText(text),
  });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!isForegroundMessage(message)) return undefined;
    sendResponse(controller.handleMessage(message));
    return undefined;
  });

  registry[INSTALLED] = controller;
  return controller;
}
