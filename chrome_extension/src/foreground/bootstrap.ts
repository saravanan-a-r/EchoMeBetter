/**
 * Wires the foreground controller to Chrome. Idempotent: the service worker
 * injects this script on every menu click, and only the first injection in
 * a frame installs anything.
 */
import { isForegroundMessage, JOB_PORT_NAME, SETUP_PORT_NAME, type AccessOfferReply, type AccessRequest, type AccessResult, type OpenSettingsRequest, type ReplacedNotice, type StyleMenuRequest } from '../shared/messages';
import { ForegroundController, type JobPortLike, type SetupPortLike } from './controller';
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
    connectSetup: () => chrome.runtime.connect({ name: SETUP_PORT_NAME }) as unknown as SetupPortLike,
    overlay: () => {
      // Re-mount if the page removed our host element (some apps own all of <html>).
      if (!overlay || !overlay.host.isConnected) overlay = mountOverlay(win.document);
      return overlay.store;
    },
    writeClipboard: (text) => win.navigator.clipboard.writeText(text),
    openSettings: (focus) => {
      const request: OpenSettingsRequest = { kind: 'page/open-settings', focus };
      chrome.runtime.sendMessage(request).catch(() => undefined);
    },
    requestRewrite: (style) => {
      const request: StyleMenuRequest = { kind: 'menu/rewrite', style };
      chrome.runtime.sendMessage(request).catch(() => undefined);
    },
    siteAccess: {
      afterReplace: async () => {
        const notice: ReplacedNotice = { kind: 'access/replaced' };
        return ((await chrome.runtime.sendMessage(notice)) as AccessOfferReply | undefined) ?? { offer: false };
      },
      request: async () => {
        const request: AccessRequest = { kind: 'access/request' };
        return (((await chrome.runtime.sendMessage(request)) as { result?: AccessResult } | undefined)?.result ?? 'denied');
      },
    },
  });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!isForegroundMessage(message)) return undefined;
    sendResponse(controller.handleMessage(message));
    return undefined;
  });

  registry[INSTALLED] = controller;
  return controller;
}
