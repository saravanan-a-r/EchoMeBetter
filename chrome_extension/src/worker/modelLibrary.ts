/**
 * The model on this device, as the inference worker manages it: is it
 * installed, download it, remove it.
 *
 * Every change is reported as an EngineEvent ('installed' and 'download'),
 * which the service worker mirrors for the popup and welcome page. One
 * download runs at a time; asking again while one runs does nothing.
 */
import { toErrorPayload, EchoError } from '../shared/errors';
import type { EngineEvent } from '../shared/messages';
import type { InstalledModelRecord } from '../shared/modelInstall';
import { summarize } from './engineHost';
import { downloadModel, type DownloadTransport } from './modelDownloader';
import { ModelStore, type StoredModel } from './modelStore';

export interface LibraryDeps {
  readonly openStore: () => Promise<ModelStore>;
  readonly transport: DownloadTransport;
  /** The loaded engine, which must let go of the model before its files go. */
  readonly engine: { unload(): Promise<void> };
  readonly emit: (event: EngineEvent) => void;
}

function toRecord(model: StoredModel): InstalledModelRecord {
  return { sourceUrl: model.sourceUrl, model: summarize(model.manifest) };
}

export class ModelLibrary {
  private store: Promise<ModelStore> | null = null;
  private download: AbortController | null = null;
  private removing = false;

  constructor(
    private readonly sourceUrl: string,
    private readonly deps: LibraryDeps,
  ) {}

  /** Report what is installed; called when the worker starts. */
  async inspect(): Promise<void> {
    const model = await this.current();
    this.deps.emit({ type: 'installed', installed: model ? toRecord(model) : null });
  }

  /** The installed model and its store, for the engine to load. */
  async require(): Promise<{ model: StoredModel; store: ModelStore }> {
    const model = this.removing ? null : await this.current();
    if (!model) {
      // The files may have gone behind the mirror's back (the browser can evict site data).
      this.deps.emit({ type: 'installed', installed: null });
      throw new EchoError('MODEL_NOT_DOWNLOADED');
    }
    return { model, store: await this.openStore() };
  }

  async startDownload(): Promise<void> {
    if (this.download || this.removing) return;
    const controller = new AbortController();
    this.download = controller;
    const emitDownload = (download: Extract<EngineEvent, { type: 'download' }>['download']) => this.deps.emit({ type: 'download', download });
    emitDownload({ state: 'downloading', phase: 'fetching', receivedBytes: 0, totalBytes: 0, bytesPerSecond: 0 });
    try {
      const existing = await this.current();
      const model =
        existing ??
        (await downloadModel(await this.openStore(), this.sourceUrl, this.deps.transport, {
          signal: controller.signal,
          onProgress: (progress) => emitDownload({ state: 'downloading', ...progress }),
        }));
      this.deps.emit({ type: 'installed', installed: toRecord(model) });
      emitDownload({ state: 'idle' });
    } catch (error) {
      if (error instanceof EchoError && error.code === 'CANCELLED') {
        // Best effort: files left behind are reused or cleaned up by the next download.
        await this.discardPartialFiles().catch(() => undefined);
        emitDownload({ state: 'idle' });
      } else {
        // Partial files stay on disk: trying again resumes them.
        emitDownload({ state: 'failed', error: toErrorPayload(error, 'DOWNLOAD_FAILED') });
      }
    } finally {
      this.download = null;
    }
  }

  cancelDownload(): void {
    this.download?.abort();
  }

  /** Delete the model; a rewrite in progress finishes first. */
  async remove(): Promise<void> {
    if (this.download || this.removing) return;
    this.removing = true;
    try {
      await this.deps.engine.unload();
      await (await this.openStore()).clear();
      this.deps.emit({ type: 'installed', installed: null });
    } finally {
      this.removing = false;
    }
  }

  private openStore(): Promise<ModelStore> {
    if (!this.store) {
      this.store = this.deps.openStore().catch((error: unknown) => {
        this.store = null;
        throw error;
      });
    }
    return this.store;
  }

  /** The installed model, if it came from the folder this build downloads from. */
  private async current(): Promise<StoredModel | null> {
    const model = await (await this.openStore()).installed();
    return model && model.sourceUrl === this.sourceUrl ? model : null;
  }

  /** After a cancel: keep whatever is installed, delete everything else. */
  private async discardPartialFiles(): Promise<void> {
    const store = await this.openStore();
    const installed = await store.installed();
    await store.prune(installed ? ModelStore.namesOf(installed.manifest) : new Set());
  }
}
