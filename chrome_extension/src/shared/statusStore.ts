/**
 * Read/write the engine status in `chrome.storage.session`. The service
 * worker writes; extension pages read and subscribe.
 */
import { INITIAL_STATUS, isEngineStatus, STATUS_STORAGE_KEY, type EngineStatus } from './status';

export async function writeStatus(status: EngineStatus, storage: chrome.storage.StorageArea = chrome.storage.session): Promise<void> {
  await storage.set({ [STATUS_STORAGE_KEY]: status });
}

export async function readStatus(storage: chrome.storage.StorageArea = chrome.storage.session): Promise<EngineStatus> {
  const stored = await storage.get(STATUS_STORAGE_KEY);
  const value = stored[STATUS_STORAGE_KEY];
  return isEngineStatus(value) ? value : INITIAL_STATUS;
}

export function subscribeStatus(
  listener: (status: EngineStatus) => void,
  events: Pick<typeof chrome.storage.onChanged, 'addListener' | 'removeListener'> = chrome.storage.onChanged,
): () => void {
  const handler = (changes: Record<string, chrome.storage.StorageChange>, area: string) => {
    if (area !== 'session' || !(STATUS_STORAGE_KEY in changes)) return;
    const next = changes[STATUS_STORAGE_KEY]!.newValue;
    listener(isEngineStatus(next) ? next : INITIAL_STATUS);
  };
  events.addListener(handler);
  return () => events.removeListener(handler);
}
