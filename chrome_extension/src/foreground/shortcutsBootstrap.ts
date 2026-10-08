/**
 * Wires the page listener to Chrome: the style shortcuts and the press and
 * hold that opens the style menu, each on while its setting is. Idempotent:
 * the listener is both registered as a content script and injected into
 * tabs that were already open, so a frame can be reached twice.
 */
import type { ShortcutRequest, StyleMenuRequest } from '../shared/messages';
import { DEFAULT_SETTINGS, loadSettings, parseSettings, SETTINGS_STORAGE_KEY, type Settings } from '../shared/settings';
import { detectPlatform } from '../shared/shortcuts';
import { leaveHeldSelection } from './heldSelection';
import { listenForHold } from './holdListener';
import { listenForShortcuts } from './shortcutListener';

const INSTALLED = Symbol.for('echomebetter.shortcuts');

function runtimeAlive(): boolean {
  try {
    return typeof chrome !== 'undefined' && Boolean(chrome.runtime?.id);
  } catch {
    return false;
  }
}

function send(request: ShortcutRequest | StyleMenuRequest): void {
  chrome.runtime.sendMessage(request).catch(() => undefined);
}

export function installShortcuts(win: Window = window): void {
  const registry = win as unknown as Record<symbol, boolean | undefined>;
  if (registry[INSTALLED]) return;
  registry[INSTALLED] = true;

  let settings: Pick<Settings, 'shortcutsEnabled' | 'holdMenuEnabled'> = DEFAULT_SETTINGS;
  loadSettings()
    .then((stored) => (settings = stored))
    .catch(() => undefined);
  chrome.storage.onChanged.addListener((changes, area) => {
    const change = changes[SETTINGS_STORAGE_KEY];
    if (area === 'local' && change) settings = parseSettings(change.newValue);
  });

  listenForShortcuts({
    win,
    platform: detectPlatform(win.navigator as Navigator & { userAgentData?: { platform?: string } }),
    // After the extension is reloaded or removed, this listener is left behind and must stay out of the way.
    enabled: () => settings.shortcutsEnabled && runtimeAlive(),
    trigger: (style) => send({ kind: 'shortcut/rewrite', style }),
  });

  listenForHold({
    win,
    enabled: () => settings.holdMenuEnabled && runtimeAlive(),
    open: (point, target) => {
      leaveHeldSelection(win, point, target);
      send({ kind: 'menu/open', point });
    },
  });
}
