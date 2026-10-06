/**
 * What is on this device, what can be downloaded, and is a download running?
 *
 *   installed model   chrome.storage.local    lasts as long as the files do
 *   download          chrome.storage.session  a download never outlives the browser session
 *
 * The model is a base plus one adapter per style; the base comes first, and
 * any set of adapters can be added to it or removed again. The files live in
 * the inference worker's private file system, which is the truth; the worker
 * reports every change and the service worker mirrors it here, so the popup
 * and welcome page render instantly without starting the worker.
 */
import { isErrorPayload, type ErrorPayload } from './errors';
import type { ModelSummary } from './status';
import { isStyleId, STYLE_IDS, styleLabel, type StyleId } from './styles';

export interface AdapterSummary {
  /** Adapters are named after the style they were trained for. */
  readonly style: StyleId;
  readonly sizeBytes: number;
}

/** What one model.json offers. */
export interface ModelCatalog {
  /** Identifies the base weights; adapters only fit the base they were trained on. */
  readonly base: string;
  readonly model: ModelSummary;
  readonly adapters: readonly AdapterSummary[];
  /** The adapter styles without one of their own run with. */
  readonly fallbackAdapter: StyleId | null;
}

export interface InstalledModelRecord {
  /** The folder it was downloaded from; a build pointing elsewhere needs a new download. */
  readonly sourceUrl: string;
  /** What the installed model.json offers. */
  readonly catalog: ModelCatalog;
  /** The adapters on this device. */
  readonly adapters: readonly StyleId[];
}

export interface DownloadTarget {
  /** The base is downloaded along with the first adapters. */
  readonly base: boolean;
  readonly adapters: readonly StyleId[];
}

export type DownloadState =
  | { readonly state: 'idle' }
  | {
      readonly state: 'downloading';
      readonly target: DownloadTarget;
      /** Fetching the files, then checking each one against its sha256. */
      readonly phase: 'fetching' | 'verifying';
      readonly receivedBytes: number;
      /** 0 until model.json has been read. */
      readonly totalBytes: number;
      readonly bytesPerSecond: number;
    }
  | { readonly state: 'failed'; readonly target: DownloadTarget; readonly error: ErrorPayload };

export const INSTALLED_MODEL_KEY = 'installedModel';
export const DOWNLOAD_STATE_KEY = 'modelDownload';
export const IDLE_DOWNLOAD: DownloadState = { state: 'idle' };

function isStyleList(value: unknown): value is StyleId[] {
  return Array.isArray(value) && value.every(isStyleId);
}

function isModelCatalog(value: unknown): value is ModelCatalog {
  if (typeof value !== 'object' || value === null) return false;
  const catalog = value as Partial<Record<keyof ModelCatalog, unknown>>;
  return (
    typeof catalog.base === 'string' &&
    typeof catalog.model === 'object' &&
    catalog.model !== null &&
    Array.isArray(catalog.adapters) &&
    catalog.adapters.every((adapter: { style?: unknown; sizeBytes?: unknown }) => isStyleId(adapter.style) && typeof adapter.sizeBytes === 'number') &&
    (catalog.fallbackAdapter === null || isStyleId(catalog.fallbackAdapter))
  );
}

export function isInstalledModelRecord(value: unknown): value is InstalledModelRecord {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as { sourceUrl?: unknown; catalog?: unknown; adapters?: unknown };
  return typeof record.sourceUrl === 'string' && isModelCatalog(record.catalog) && isStyleList(record.adapters);
}

function isDownloadTarget(value: unknown): value is DownloadTarget {
  if (typeof value !== 'object' || value === null) return false;
  const target = value as { base?: unknown; adapters?: unknown };
  return typeof target.base === 'boolean' && isStyleList(target.adapters);
}

export function isDownloadState(value: unknown): value is DownloadState {
  if (typeof value !== 'object' || value === null) return false;
  const download = value as { state?: unknown; error?: unknown; target?: unknown };
  if (download.state === 'idle') return true;
  if (!isDownloadTarget(download.target)) return false;
  return download.state === 'downloading' || (download.state === 'failed' && isErrorPayload(download.error));
}

/** The installed model, provided it came from the folder this build downloads from. */
export function modelFrom(installed: InstalledModelRecord | null, sourceUrl: string): InstalledModelRecord | null {
  return installed && installed.sourceUrl === sourceUrl ? installed : null;
}

/** The adapter `style` runs with under `catalog`: its own, else the fallback, else none. */
export function adapterForStyle(catalog: ModelCatalog, style: StyleId): StyleId | null {
  return catalog.adapters.some((adapter) => adapter.style === style) ? style : catalog.fallbackAdapter;
}

/** The styles that run with `adapter` under `catalog`, in menu order. */
export function stylesUsing(catalog: ModelCatalog, adapter: StyleId): StyleId[] {
  return STYLE_IDS.filter((style) => adapterForStyle(catalog, style) === adapter);
}

/** Why `style` cannot be used right now, or null when it can. */
export function styleProblem(style: StyleId, installed: InstalledModelRecord | null): ErrorPayload | null {
  if (!installed) return { code: 'MODEL_NOT_DOWNLOADED' };
  const adapter = adapterForStyle(installed.catalog, style);
  if (!adapter) return { code: 'STYLE_UNAVAILABLE', details: { style: styleLabel(style) } };
  if (!installed.adapters.includes(adapter)) return { code: 'STYLE_NOT_DOWNLOADED', details: { style: styleLabel(style), adapter: styleLabel(adapter) } };
  return null;
}

/** Whether any style can be used: the base and at least one adapter are on this device. */
export function canRewrite(installed: InstalledModelRecord | null): boolean {
  return STYLE_IDS.some((style) => styleProblem(style, installed) === null);
}

/**
 * The catalog the pages offer downloads from: the server's when it fits the
 * installed base (it may list adapters trained since), else the installed one.
 */
export function offeredCatalog(installed: InstalledModelRecord | null, remote: ModelCatalog | null): ModelCatalog | null {
  if (!installed) return remote;
  return remote && remote.base === installed.catalog.base ? remote : installed.catalog;
}

export type StyleReadiness =
  /** Nothing to do: the style rewrites with `adapter`. */
  | { readonly state: 'ready'; readonly adapter: StyleId }
  /** `adapter` has to be downloaded first. */
  | { readonly state: 'needs-download'; readonly adapter: AdapterSummary }
  /** The base model has to be downloaded first. */
  | { readonly state: 'needs-model'; readonly adapter: AdapterSummary }
  | { readonly state: 'unavailable' };

/** Where `style` stands, given what is offered and what is installed. */
export function styleReadiness(style: StyleId, catalog: ModelCatalog, installed: InstalledModelRecord | null): StyleReadiness {
  const id = adapterForStyle(catalog, style);
  const adapter = catalog.adapters.find((candidate) => candidate.style === id);
  if (!adapter) return { state: 'unavailable' };
  if (!installed) return { state: 'needs-model', adapter };
  return installed.adapters.includes(adapter.style) ? { state: 'ready', adapter: adapter.style } : { state: 'needs-download', adapter };
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
