import { useCallback, useEffect, useState } from 'react';
import { catalogOf, parseModelManifest } from '../../engine/manifest';
import type { UiReply, UiRequest } from '../../shared/messages';
import {
  IDLE_DOWNLOAD,
  modelFrom,
  offeredCatalog,
  readDownloadState,
  readInstalledModel,
  styleReadiness,
  subscribeModelState,
  type DownloadState,
  type InstalledModelRecord,
  type ModelCatalog,
  type StyleReadiness,
} from '../../shared/modelInstall';
import { MODEL_SOURCE_URL } from '../../shared/modelSource';
import { STYLE_IDS, type StyleId } from '../../shared/styles';

export interface ModelAvailability {
  /** False until storage has been read once, so the UI does not flash the download card. */
  readonly known: boolean;
  /** The downloaded model, if it came from the folder this build downloads from. */
  readonly installed: InstalledModelRecord | null;
  readonly download: DownloadState;
}

const UNKNOWN: ModelAvailability = { known: false, installed: null, download: IDLE_DOWNLOAD };

/** Whether the model is downloaded and how a download is going, kept current. */
export function useModelAvailability(): ModelAvailability {
  const [availability, setAvailability] = useState<ModelAvailability>(UNKNOWN);
  useEffect(() => {
    let alive = true;
    const unsubscribe = subscribeModelState({
      installed: (record) => setAvailability((current) => ({ ...current, installed: modelFrom(record, MODEL_SOURCE_URL) })),
      download: (download) => setAvailability((current) => ({ ...current, download })),
    });
    void Promise.all([readInstalledModel(), readDownloadState()]).then(([record, download]) => {
      if (alive) setAvailability({ known: true, installed: modelFrom(record, MODEL_SOURCE_URL), download });
    });
    return () => {
      alive = false;
      unsubscribe();
    };
  }, []);
  return availability;
}

/** What the download server offers right now. */
export type CatalogState = { readonly status: 'loading' } | { readonly status: 'ready'; readonly catalog: ModelCatalog } | { readonly status: 'failed' };

async function fetchCatalog(fetchImpl: typeof fetch = fetch): Promise<ModelCatalog> {
  const response = await fetchImpl(new URL('model.json', MODEL_SOURCE_URL), { cache: 'no-store' });
  if (!response.ok) throw new Error(`model.json: HTTP ${response.status}`);
  return catalogOf(parseModelManifest(await response.json()));
}

/** The server's catalog, read when the page opens; `retry` reads it again. */
export function useCatalog(): [CatalogState, () => void] {
  const [state, setState] = useState<CatalogState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let alive = true;
    setState({ status: 'loading' });
    fetchCatalog().then(
      (catalog) => alive && setState({ status: 'ready', catalog }),
      () => alive && setState({ status: 'failed' }),
    );
    return () => {
      alive = false;
    };
  }, [attempt]);
  return [state, useCallback(() => setAttempt((count) => count + 1), [])];
}

export interface ModelSetup {
  /** What downloads are offered from: the server's catalog, or the installed one when the server is out of reach. */
  readonly catalog: ModelCatalog | null;
  readonly readiness: Readonly<Partial<Record<StyleId, StyleReadiness>>>;
}

export function modelSetup(installed: InstalledModelRecord | null, remote: CatalogState): ModelSetup {
  const catalog = offeredCatalog(installed, remote.status === 'ready' ? remote.catalog : null);
  const readiness: Partial<Record<StyleId, StyleReadiness>> = {};
  if (catalog) for (const style of STYLE_IDS) readiness[style] = styleReadiness(style, catalog, installed);
  return { catalog, readiness };
}

function send(request: UiRequest): Promise<UiReply> {
  return chrome.runtime.sendMessage(request) as Promise<UiReply>;
}

/** Download `adapters`, and the base first if it is not installed yet. */
export function requestDownload(adapters: readonly StyleId[]): void {
  void send({ kind: 'ui/download-model', adapters });
}

/** Stop a running download, or put a failed one aside. */
export function requestCancelDownload(): void {
  void send({ kind: 'ui/cancel-download' });
}

export function requestRemoveModel(): Promise<UiReply> {
  return send({ kind: 'ui/remove-model' });
}

export function requestRemoveAdapter(adapter: StyleId): Promise<UiReply> {
  return send({ kind: 'ui/remove-adapter', adapter });
}
