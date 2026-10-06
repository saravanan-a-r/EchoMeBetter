/**
 * Service worker entry: every Chrome event listener is registered here,
 * synchronously at top level (MV3 only delivers events to listeners that
 * exist when the worker starts).
 */
import { toErrorPayload } from '../shared/errors';
import { isOffscreenEventMessage, isShortcutRequest, isUiRequest, JOB_PORT_NAME, type EngineEvent, type OffscreenCommand, type UiReply, type UiRequest } from '../shared/messages';
import { modelFrom, readDownloadState, readInstalledModel, styleProblem, writeDownloadState, writeInstalledModel } from '../shared/modelInstall';
import { MODEL_SOURCE_URL } from '../shared/modelSource';
import { loadSettings, parseSettings, SETTINGS_STORAGE_KEY } from '../shared/settings';
import { platformFromOs, shortcutLabel } from '../shared/shortcuts';
import type { StyleId } from '../shared/styles';
import { writeStatus } from '../shared/statusStore';
import { registerContextMenus } from './contextMenus';
import { IDLE_ALARM, lastActivity, recordActivity, shouldUnload } from './idleUnloader';
import { showInstallState } from './installBadge';
import { JobRouter } from './jobRouter';
import { OffscreenManager } from './offscreenManager';
import { launchRewrite, launchStyle, type LauncherDeps } from './rewriteLauncher';
import { shortcutsActive, syncShortcutScript } from './shortcutAccess';

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

async function checkStyle(style: StyleId) {
  return styleProblem(style, modelFrom(await readInstalledModel(), MODEL_SOURCE_URL));
}

async function refreshInstallBadge(): Promise<void> {
  await showInstallState(await readInstalledModel(), MODEL_SOURCE_URL);
}

function sendModelCommand(command: Extract<OffscreenCommand, { kind: `model/${string}` }>): Promise<void> {
  return ensureEngineHost().then(() => offscreen.send(command));
}

/** Stop a running download; a failed one (or a stale state with no worker behind it) is simply put aside. */
async function cancelDownload(): Promise<void> {
  const running = (await readDownloadState()).state === 'downloading' && (await offscreen.exists());
  if (running) await offscreen.send({ target: 'offscreen', kind: 'model/cancel-download' });
  else await writeDownloadState({ state: 'idle' });
}

function handleUiRequest(request: UiRequest): UiReply {
  switch (request.kind) {
    case 'ui/warm-up':
      prepareEngine();
      return { ok: true };
    case 'ui/download-model':
      sendModelCommand({ target: 'offscreen', kind: 'model/download', adapters: request.adapters }).catch(async (error: unknown) => {
        const base = modelFrom(await readInstalledModel(), MODEL_SOURCE_URL) === null;
        await writeDownloadState({ state: 'failed', target: { base, adapters: request.adapters }, error: toErrorPayload(error, 'DOWNLOAD_FAILED') });
      });
      return { ok: true };
    case 'ui/cancel-download':
      void cancelDownload().catch(() => writeDownloadState({ state: 'idle' }));
      return { ok: true };
    case 'ui/remove-adapter':
      if (router.activeJobCount > 0) return { ok: false, error: { code: 'BUSY' } };
      sendModelCommand({ target: 'offscreen', kind: 'model/remove-adapter', adapter: request.adapter }).catch((error: unknown) =>
        console.error('EchoMeBetter: could not remove the style', error),
      );
      return { ok: true };
    case 'ui/remove-model':
      if (router.activeJobCount > 0) return { ok: false, error: { code: 'BUSY' } };
      sendModelCommand({ target: 'offscreen', kind: 'model/remove' }).catch((error: unknown) => console.error('EchoMeBetter: could not remove the model', error));
      return { ok: true };
  }
}

async function mirrorEngineEvent(event: EngineEvent): Promise<void> {
  switch (event.type) {
    case 'status':
      await writeStatus(event.status);
      // A crashed worker takes a running download with it.
      if (event.status.state === 'error') {
        const download = await readDownloadState();
        if (download.state === 'downloading') {
          await writeDownloadState({ state: 'failed', target: download.target, error: { code: 'DOWNLOAD_FAILED', details: { reason: event.status.message } } });
        }
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

const launcher: LauncherDeps = {
  scripting: chrome.scripting,
  tabs: chrome.tabs,
  action: chrome.action,
  extensionOrigin: chrome.runtime.getURL(''),
  prepareEngine,
  checkStyle,
  newJobId: () => crypto.randomUUID(),
};

async function shortcutsOn(): Promise<boolean> {
  return shortcutsActive((await loadSettings()).shortcutsEnabled, chrome.permissions);
}

async function applyShortcuts(): Promise<void> {
  const active = await shortcutsOn();
  await syncShortcutScript(active, { scripting: chrome.scripting, permissions: chrome.permissions, tabs: chrome.tabs });
  const platform = platformFromOs((await chrome.runtime.getPlatformInfo()).os);
  await registerContextMenus(chrome.contextMenus, (style) => (active ? shortcutLabel(style, platform) : null));
}

// One at a time: rebuilding the menu while another rebuild runs would create duplicate ids.
let shortcutsApplied: Promise<void> = Promise.resolve();
function refreshShortcuts(): void {
  shortcutsApplied = shortcutsApplied.then(applyShortcuts).catch((error: unknown) => console.error('EchoMeBetter: could not update shortcuts', error));
}

async function handleShortcut(style: StyleId, sender: chrome.runtime.MessageSender): Promise<void> {
  const tab = sender.tab;
  if (!tab) return;
  // The welcome page listens on its own; websites only while shortcuts are on.
  const ownPage = tab.url?.startsWith(launcher.extensionOrigin) ?? false;
  if (!ownPage && !(await shortcutsOn())) return;
  if (ownPage && !(await loadSettings()).shortcutsEnabled) return;
  await launchStyle(style, sender.frameId ?? 0, tab, launcher);
}

const router = new JobRouter({
  ensureEngineHost,
  sendToEngine: (command) => offscreen.send(command),
  onActivity: () => void recordActivity(),
});

chrome.runtime.onInstalled.addListener((details) => {
  refreshShortcuts();
  void refreshInstallBadge();
  if (details.reason === chrome.runtime.OnInstalledReason.INSTALL) {
    void chrome.tabs.create({ url: chrome.runtime.getURL(WELCOME_PAGE) });
  }
});

// The badge is not kept across browser restarts.
chrome.runtime.onStartup.addListener(() => void refreshInstallBadge());

chrome.contextMenus.onClicked.addListener((info, tab) => {
  void launchRewrite(info, tab, launcher);
});

chrome.permissions.onAdded.addListener(refreshShortcuts);
chrome.permissions.onRemoved.addListener(refreshShortcuts);
chrome.storage.onChanged.addListener((changes, area) => {
  const change = changes[SETTINGS_STORAGE_KEY];
  if (area !== 'local' || !change) return;
  if (parseSettings(change.oldValue).shortcutsEnabled !== parseSettings(change.newValue).shortcutsEnabled) refreshShortcuts();
});

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === JOB_PORT_NAME) router.handlePort(port);
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (isOffscreenEventMessage(message)) {
    router.handleEngineEvent(message.event);
    void mirrorEngineEvent(message.event);
    return undefined;
  }
  if (isShortcutRequest(message)) {
    void handleShortcut(message.style, sender);
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
