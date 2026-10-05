/**
 * Service worker entry: every Chrome event listener is registered here,
 * synchronously at top level (MV3 only delivers events to listeners that
 * exist when the worker starts).
 */
import { toErrorPayload } from '../shared/errors';
import { isOffscreenEventMessage, isUiRequest, JOB_PORT_NAME, type EngineEvent, type OffscreenCommand, type UiReply, type UiRequest } from '../shared/messages';
import { modelFrom, readDownloadState, readInstalledModel, writeDownloadState, writeInstalledModel } from '../shared/modelInstall';
import { MODEL_SOURCE_URL } from '../shared/modelSource';
import { loadSettings } from '../shared/settings';
import { writeStatus } from '../shared/statusStore';
import { registerContextMenus } from './contextMenus';
import { IDLE_ALARM, lastActivity, recordActivity, shouldUnload } from './idleUnloader';
import { showInstallState } from './installBadge';
import { JobRouter } from './jobRouter';
import { OffscreenManager } from './offscreenManager';
import { launchRewrite } from './rewriteLauncher';

const WELCOME_PAGE = 'ui/welcome/welcome.html';

const offscreen = new OffscreenManager();

async function ensureEngineHost(): Promise<void> {
  await recordActivity();
  await offscreen.ensure();
  await chrome.alarms.create(IDLE_ALARM, { periodInMinutes: 1 });
}

function prepareEngine(): void {
  ensureEngineHost()
    .then(() => offscreen.send({ target: 'offscreen', kind: 'engine/warm-up' }))
    .catch((error: unknown) => writeStatus({ state: 'error', message: error instanceof Error ? error.message : String(error) }));
}

async function modelReady(): Promise<boolean> {
  return modelFrom(await readInstalledModel(), MODEL_SOURCE_URL) !== null;
}

async function refreshInstallBadge(): Promise<void> {
  await showInstallState(await readInstalledModel(), MODEL_SOURCE_URL);
}

function sendModelCommand(kind: Extract<OffscreenCommand['kind'], `model/${string}`>): Promise<void> {
  return ensureEngineHost().then(() => offscreen.send({ target: 'offscreen', kind }));
}

function handleUiRequest(request: UiRequest): UiReply {
  switch (request.kind) {
    case 'ui/warm-up':
      prepareEngine();
      return { ok: true };
    case 'ui/download-model':
      sendModelCommand('model/download').catch((error: unknown) =>
        writeDownloadState({ state: 'failed', error: toErrorPayload(error, 'DOWNLOAD_FAILED') }),
      );
      return { ok: true };
    case 'ui/cancel-download':
      // Without the offscreen document no download can be running; only a stale state needs clearing.
      void offscreen
        .exists()
        .then((running) => (running ? offscreen.send({ target: 'offscreen', kind: 'model/cancel-download' }) : writeDownloadState({ state: 'idle' })))
        .catch(() => writeDownloadState({ state: 'idle' }));
      return { ok: true };
    case 'ui/remove-model':
      if (router.activeJobCount > 0) return { ok: false, error: { code: 'BUSY' } };
      sendModelCommand('model/remove').catch((error: unknown) => console.error('EchoMeBetter: could not remove the model', error));
      return { ok: true };
  }
}

async function mirrorEngineEvent(event: EngineEvent): Promise<void> {
  switch (event.type) {
    case 'status':
      await writeStatus(event.status);
      // A crashed worker takes a running download with it.
      if (event.status.state === 'error' && (await readDownloadState()).state === 'downloading') {
        await writeDownloadState({ state: 'failed', error: { code: 'DOWNLOAD_FAILED', details: { reason: event.status.message } } });
      }
      return;
    case 'installed':
      await writeInstalledModel(event.installed);
      await showInstallState(event.installed, MODEL_SOURCE_URL);
      return;
    case 'download':
      await writeDownloadState(event.download);
      return;
    default:
      return;
  }
}

const router = new JobRouter({
  ensureEngineHost,
  sendToEngine: (command) => offscreen.send(command),
  onActivity: () => void recordActivity(),
});

chrome.runtime.onInstalled.addListener((details) => {
  void registerContextMenus();
  void refreshInstallBadge();
  if (details.reason === chrome.runtime.OnInstalledReason.INSTALL) {
    void chrome.tabs.create({ url: chrome.runtime.getURL(WELCOME_PAGE) });
  }
});

// The badge is not kept across browser restarts.
chrome.runtime.onStartup.addListener(() => void refreshInstallBadge());

chrome.contextMenus.onClicked.addListener((info, tab) => {
  void launchRewrite(info, tab, {
    scripting: chrome.scripting,
    tabs: chrome.tabs,
    action: chrome.action,
    extensionOrigin: chrome.runtime.getURL(''),
    prepareEngine,
    modelReady,
    newJobId: () => crypto.randomUUID(),
  });
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === JOB_PORT_NAME) router.handlePort(port);
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (isOffscreenEventMessage(message)) {
    router.handleEngineEvent(message.event);
    void mirrorEngineEvent(message.event);
    return undefined;
  }
  if (isUiRequest(message)) sendResponse(handleUiRequest(message));
  return undefined;
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name !== IDLE_ALARM) return;
  void (async () => {
    if (!(await offscreen.exists())) {
      await chrome.alarms.clear(IDLE_ALARM);
      return;
    }
    const unload = shouldUnload({
      now: Date.now(),
      lastActivityAt: await lastActivity(),
      activeJobs: router.activeJobCount,
      downloading: (await readDownloadState()).state === 'downloading',
      settings: await loadSettings(),
    });
    if (!unload) return;
    await offscreen.close();
    await chrome.alarms.clear(IDLE_ALARM);
    await writeStatus({ state: 'unloaded' });
  })();
});
