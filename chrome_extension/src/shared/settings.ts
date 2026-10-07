/**
 * User preferences, persisted in `chrome.storage.local`.
 *
 * The setting that matters most for "never slow the browser down": how long
 * an idle model stays in memory. Loaded, the int8 model holds roughly a
 * gigabyte; unloading it costs a few seconds on the next rewrite. Where the
 * model runs and how much of the processor it may use are in shared/compute.ts.
 */
import { CPU_USAGES, DEFAULT_COMPUTE, GPU_POWERS, PROCESSORS, type ComputeSettings } from './compute';

export const KEEP_LOADED_CHOICES = [5, 15, 60, 0] as const; // minutes; 0 = until the browser closes

export type KeepLoadedMinutes = (typeof KEEP_LOADED_CHOICES)[number];

export interface Settings extends ComputeSettings {
  readonly keepModelLoadedMinutes: KeepLoadedMinutes;
  /** Keyboard shortcuts on web pages; they also need the optional site access (see shared/shortcuts.ts). */
  readonly shortcutsEnabled: boolean;
}

export const DEFAULT_SETTINGS: Settings = { keepModelLoadedMinutes: 15, shortcutsEnabled: true, ...DEFAULT_COMPUTE };

export const SETTINGS_STORAGE_KEY = 'settings';

function oneOf<T>(choices: readonly T[], value: unknown, fallback: T): T {
  return (choices as readonly unknown[]).includes(value) ? (value as T) : fallback;
}

/** Accept only known values; anything else (old versions, tampering) falls back to the default. */
export function parseSettings(raw: unknown): Settings {
  if (typeof raw !== 'object' || raw === null) return DEFAULT_SETTINGS;
  const stored = raw as Partial<Record<keyof Settings, unknown>>;
  return {
    keepModelLoadedMinutes: oneOf(KEEP_LOADED_CHOICES, stored.keepModelLoadedMinutes, DEFAULT_SETTINGS.keepModelLoadedMinutes),
    shortcutsEnabled: typeof stored.shortcutsEnabled === 'boolean' ? stored.shortcutsEnabled : DEFAULT_SETTINGS.shortcutsEnabled,
    processor: oneOf(PROCESSORS, stored.processor, DEFAULT_SETTINGS.processor),
    cpuUsage: oneOf(CPU_USAGES, stored.cpuUsage, DEFAULT_SETTINGS.cpuUsage),
    gpuPower: oneOf(GPU_POWERS, stored.gpuPower, DEFAULT_SETTINGS.gpuPower),
  };
}

export async function loadSettings(storage: chrome.storage.StorageArea = chrome.storage.local): Promise<Settings> {
  const stored = await storage.get(SETTINGS_STORAGE_KEY);
  return parseSettings(stored[SETTINGS_STORAGE_KEY]);
}

export async function saveSettings(settings: Settings, storage: chrome.storage.StorageArea = chrome.storage.local): Promise<void> {
  await storage.set({ [SETTINGS_STORAGE_KEY]: parseSettings(settings) });
}
