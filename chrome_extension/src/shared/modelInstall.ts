/**
 * Is the model on this device, and is a download running?
 *
 *   installed model   chrome.storage.local    lasts as long as the files do
 *   download          chrome.storage.session  a download never outlives the browser session
 *
 * The files themselves live in the inference worker's private file system,
 * which is the truth; the worker reports every change and the service worker
 * mirrors it here, so the popup and welcome page render instantly without
 * starting the worker.
 */
import { isErrorPayload, type ErrorPayload } from './errors';
import type { ModelSummary } from './status';

export interface InstalledModelRecord {
  /** The folder it was downloaded from; a build pointing elsewhere needs a new download. */
  readonly sourceUrl: string;
  readonly model: ModelSummary;
}

export type DownloadState =
  | { readonly state: 'idle' }
  | {
      readonly state: 'downloading';
      /** Fetching the files, then checking each one against its sha256. */
      readonly phase: 'fetching' | 'verifying';
      readonly receivedBytes: number;
      /** 0 until model.json has been read. */
      readonly totalBytes: number;
      readonly bytesPerSecond: number;
    }
  | { readonly state: 'failed'; readonly error: ErrorPayload };

export const INSTALLED_MODEL_KEY = 'installedModel';
export const DOWNLOAD_STATE_KEY = 'modelDownload';
export const IDLE_DOWNLOAD: DownloadState = { state: 'idle' };

export function isInstalledModelRecord(value: unknown): value is InstalledModelRecord {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as { sourceUrl?: unknown; model?: unknown };
  return typeof record.sourceUrl === 'string' && typeof record.model === 'object' && record.model !== null;
}

export function isDownloadState(value: unknown): value is DownloadState {
  if (typeof value !== 'object' || value === null) return false;
  const download = value as { state?: unknown; error?: unknown };
  if (download.state === 'failed') return isErrorPayload(download.error);
  return download.state === 'idle' || download.state === 'downloading';
}

/** The installed model, provided it came from the folder this build downloads from. */
export function modelFrom(installed: InstalledModelRecord | null, sourceUrl: string): InstalledModelRecord | null {
  return installed && installed.sourceUrl === sourceUrl ? installed : null;
}

type StorageEvents = Pick<typeof chrome.storage.onChanged, 'addListener' | 'removeListener'>;

export async function writeInstalledModel(
  installed: InstalledModelRecord | null,
  storage: chrome.storage.StorageArea = chrome.storage.local,
): Promise<void> {
  if (installed) await storage.set({ [INSTALLED_MODEL_KEY]: installed });
  else await storage.remove(INSTALLED_MODEL_KEY);
}

export async function readInstalledModel(storage: chrome.storage.StorageArea = chrome.storage.local): Promise<InstalledModelRecord | null> {
  const value = (await storage.get(INSTALLED_MODEL_KEY))[INSTALLED_MODEL_KEY];
  return isInstalledModelRecord(value) ? value : null;
}

export async function writeDownloadState(download: DownloadState, storage: chrome.storage.StorageArea = chrome.storage.session): Promise<void> {
  await storage.set({ [DOWNLOAD_STATE_KEY]: download });
}

export async function readDownloadState(storage: chrome.storage.StorageArea = chrome.storage.session): Promise<DownloadState> {
  const value = (await storage.get(DOWNLOAD_STATE_KEY))[DOWNLOAD_STATE_KEY];
  return isDownloadState(value) ? value : IDLE_DOWNLOAD;
}

export function subscribeModelState(
  listener: { installed(record: InstalledModelRecord | null): void; download(state: DownloadState): void },
  events: StorageEvents = chrome.storage.onChanged,
): () => void {
  const handler = (changes: Record<string, chrome.storage.StorageChange>, area: string) => {
    if (area === 'local' && INSTALLED_MODEL_KEY in changes) {
      const next = changes[INSTALLED_MODEL_KEY]!.newValue;
      listener.installed(isInstalledModelRecord(next) ? next : null);
    }
    if (area === 'session' && DOWNLOAD_STATE_KEY in changes) {
      const next = changes[DOWNLOAD_STATE_KEY]!.newValue;
      listener.download(isDownloadState(next) ? next : IDLE_DOWNLOAD);
    }
  };
  events.addListener(handler);
  return () => events.removeListener(handler);
}
