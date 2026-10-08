/**
 * Frees the model's memory when it has not been used for a while.
 *
 * The last-activity time lives in `chrome.storage.session` rather than a
 * variable because the service worker itself is suspended between events;
 * a once-a-minute alarm wakes it to check.
 *
 * Freeing it this way is recorded too (it "rests"), until the model is next
 * loaded: the rewrite that has to wake it up then tells the user how long
 * that took, and that a longer keep-awake time would avoid it.
 */
import type { WokeFromRest } from '../shared/messages';
import type { Settings } from '../shared/settings';

export const IDLE_ALARM = 'echomebetter/idle-check';
export const ACTIVITY_STORAGE_KEY = 'lastActivityAt';
export const RESTING_STORAGE_KEY = 'modelResting';

export interface IdleInputs {
  readonly now: number;
  readonly lastActivityAt: number;
  readonly activeJobs: number;
  /** A model download runs in the same worker; closing it would stop the download. */
  readonly downloading: boolean;
  readonly settings: Pick<Settings, 'keepModelLoadedMinutes'>;
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

/** The idle timer has just freed the model. */
export async function recordRest(storage: chrome.storage.StorageArea = chrome.storage.session): Promise<void> {
  await storage.set({ [RESTING_STORAGE_KEY]: true });
}

/** The model is loaded again, by whatever means. */
export async function endRest(storage: chrome.storage.StorageArea = chrome.storage.session): Promise<void> {
  await storage.remove(RESTING_STORAGE_KEY);
}

/**
 * For a rewrite about to start: whether it will wake the model from a rest
 * that a longer keep-awake time would have avoided (none would with "Never").
 */
export async function wakingFromRest(
  settings: Pick<Settings, 'keepModelLoadedMinutes'>,
  storage: chrome.storage.StorageArea = chrome.storage.session,
): Promise<WokeFromRest | null> {
  const resting = (await storage.get(RESTING_STORAGE_KEY))[RESTING_STORAGE_KEY] === true;
  const minutes = settings.keepModelLoadedMinutes;
  return resting && minutes !== 0 ? { idleMinutes: minutes } : null;
}
