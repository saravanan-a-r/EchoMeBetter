import { useEffect, useState } from 'react';
import type { UiReply, UiRequest } from '../../shared/messages';
import {
  IDLE_DOWNLOAD,
  modelFrom,
  readDownloadState,
  readInstalledModel,
  subscribeModelState,
  type DownloadState,
  type InstalledModelRecord,
} from '../../shared/modelInstall';
import { MODEL_SOURCE_URL } from '../../shared/modelSource';

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

function send(kind: UiRequest['kind']): Promise<UiReply> {
  return chrome.runtime.sendMessage({ kind } satisfies UiRequest) as Promise<UiReply>;
}

export function requestDownload(): void {
  void send('ui/download-model');
}

export function requestCancelDownload(): void {
  void send('ui/cancel-download');
}

export function requestRemoveModel(): Promise<UiReply> {
  return send('ui/remove-model');
}
