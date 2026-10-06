/**
 * Wires the shortcut listener to Chrome. Idempotent: the listener is both
 * registered as a content script and injected into tabs that were already
 * open, so a frame can be reached twice.
 */
import type { ShortcutRequest } from '../shared/messages';
import { DEFAULT_SETTINGS, loadSettings, parseSettings, SETTINGS_STORAGE_KEY } from '../shared/settings';
import { detectPlatform } from '../shared/shortcuts';
import { listenForShortcuts } from './shortcutListener';

const INSTALLED = Symbol.for('echomebetter.shortcuts');

function runtimeAlive(): boolean {
  try {
    return typeof chrome !== 'undefined' && Boolean(chrome.runtime?.id);
  } catch {
    return false;
  }
}

export function installShortcuts(win: Window = window): void {
  const registry = win as unknown as Record<symbol, boolean | undefined>;
  if (registry[INSTALLED]) return;
  registry[INSTALLED] = true;

  let enabled = DEFAULT_SETTINGS.shortcutsEnabled;
  loadSettings()
    .then((settings) => (enabled = settings.shortcutsEnabled))
    .catch(() => undefined);
  chrome.storage.onChanged.addListener((changes, area) => {
    const change = changes[SETTINGS_STORAGE_KEY];
    if (area === 'local' && change) enabled = parseSettings(change.newValue).shortcutsEnabled;
  });

  listenForShortcuts({
    win,
    platform: detectPlatform(win.navigator as Navigator & { userAgentData?: { platform?: string } }),
    // After the extension is reloaded or removed, this listener is left behind and must stay out of the way.
    enabled: () => enabled && runtimeAlive(),
    trigger: (style) => {
      const request: ShortcutRequest = { kind: 'shortcut/rewrite', style };
      chrome.runtime.sendMessage(request).catch(() => undefined);
    },
  });
}
