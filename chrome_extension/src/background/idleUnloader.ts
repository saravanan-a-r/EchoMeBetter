/**
 * Frees the model's memory when it has not been used for a while.
 *
 * The last-activity time lives in `chrome.storage.session` rather than a
 * variable because the service worker itself is suspended between events;
 * a once-a-minute alarm wakes it to check.
 */
import type { Settings } from '../shared/settings';

export const IDLE_ALARM = 'echomebetter/idle-check';
export const ACTIVITY_STORAGE_KEY = 'lastActivityAt';

export interface IdleInputs {
  readonly now: number;
  readonly lastActivityAt: number;
  readonly activeJobs: number;
  /** A model download runs in the same worker; closing it would stop the download. */
  readonly downloading: boolean;
  readonly settings: Settings;
}

export function shouldUnload({ now, lastActivityAt, activeJobs, downloading, settings }: IdleInputs): boolean {
  const minutes = settings.keepModelLoadedMinutes;
  if (minutes === 0 || activeJobs > 0 || downloading) return false;
  return now - lastActivityAt >= minutes * 60_000;
}

export async function recordActivity(now = Date.now(), storage: chrome.storage.StorageArea = chrome.storage.session): Promise<void> {
  await storage.set({ [ACTIVITY_STORAGE_KEY]: now });
}

export async function lastActivity(storage: chrome.storage.StorageArea = chrome.storage.session): Promise<number> {
  const stored = await storage.get(ACTIVITY_STORAGE_KEY);
  const value = stored[ACTIVITY_STORAGE_KEY];
  return typeof value === 'number' ? value : 0;
}
