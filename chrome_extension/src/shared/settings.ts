/**
 * User preferences, persisted in `chrome.storage.local`.
 *
 * The setting that matters most for "never slow the browser down": how long
 * an idle model stays in memory. Loaded, the int8 model holds roughly a
 * gigabyte; unloading it costs a few seconds on the next rewrite.
 */
export const KEEP_LOADED_CHOICES = [5, 15, 60, 0] as const; // minutes; 0 = until the browser closes

export type KeepLoadedMinutes = (typeof KEEP_LOADED_CHOICES)[number];

export interface Settings {
  readonly keepModelLoadedMinutes: KeepLoadedMinutes;
  /** Keyboard shortcuts on web pages; they also need the optional site access (see shared/shortcuts.ts). */
  readonly shortcutsEnabled: boolean;
}

export const DEFAULT_SETTINGS: Settings = { keepModelLoadedMinutes: 15, shortcutsEnabled: true };

export const SETTINGS_STORAGE_KEY = 'settings';

/** Accept only known values; anything else (old versions, tampering) falls back to the default. */
export function parseSettings(raw: unknown): Settings {
  if (typeof raw !== 'object' || raw === null) return DEFAULT_SETTINGS;
  const { keepModelLoadedMinutes: minutes, shortcutsEnabled } = raw as { keepModelLoadedMinutes?: unknown; shortcutsEnabled?: unknown };
  const valid = (KEEP_LOADED_CHOICES as readonly unknown[]).includes(minutes);
  return {
    keepModelLoadedMinutes: valid ? (minutes as KeepLoadedMinutes) : DEFAULT_SETTINGS.keepModelLoadedMinutes,
    shortcutsEnabled: typeof shortcutsEnabled === 'boolean' ? shortcutsEnabled : DEFAULT_SETTINGS.shortcutsEnabled,
  };
}

export async function loadSettings(storage: chrome.storage.StorageArea = chrome.storage.local): Promise<Settings> {
  const stored = await storage.get(SETTINGS_STORAGE_KEY);
  return parseSettings(stored[SETTINGS_STORAGE_KEY]);
}

export async function saveSettings(settings: Settings, storage: chrome.storage.StorageArea = chrome.storage.local): Promise<void> {
  await storage.set({ [SETTINGS_STORAGE_KEY]: parseSettings(settings) });
}
