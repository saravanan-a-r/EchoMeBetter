/**
 * The model on this device, as the inference worker manages it: what is
 * installed, downloading the base and adapters, removing them.
 *
 * Every change is reported as an EngineEvent ('installed' and 'download'),
 * which the service worker mirrors for the popup and welcome page. One
 * download runs at a time; asking again while one runs does nothing.
 */
import { baseFingerprint, catalogOf, type ModelFile } from '../engine/manifest';
import { EchoError, toErrorPayload } from '../shared/errors';
import type { EngineEvent } from '../shared/messages';
import type { DownloadState, DownloadTarget, InstalledModelRecord } from '../shared/modelInstall';
import { STYLE_IDS, type StyleId } from '../shared/styles';
import { downloadError, downloadFiles, fetchManifest, type DownloadTransport } from './modelDownloader';
import { filesOf, ModelStore, type StoredModel } from './modelStore';

export interface LibraryDeps {
  readonly openStore: () => Promise<ModelStore>;
  readonly transport: DownloadTransport;
  /** The loaded engine, which must let go of the base before its files go. */
  readonly engine: { unload(): Promise<void> };
  readonly emit: (event: EngineEvent) => void;
}

function toRecord(model: StoredModel): InstalledModelRecord {
  return { sourceUrl: model.sourceUrl, catalog: catalogOf(model.manifest), adapters: model.adapters };
}

/**
 * The files `next` needs beyond what is installed. Installed files were
 * verified when they were downloaded; every other file is fetched (or
 * resumed, or found complete) and verified now.
 */
function filesToFetch(next: StoredModel, installed: StoredModel | null): ModelFile[] {
  const verified = installed ? ModelStore.namesOf(installed) : new Set<string>();
  const needed = filesOf(next).filter((file) => !verified.has(file.sha256));
  // Two entries with the same content are one file on disk.
  return needed.filter((file, index) => needed.findIndex((other) => other.sha256 === file.sha256) === index);
}

/** `adapters` in menu order, without repeats. */
function inStyleOrder(adapters: Iterable<StyleId>): StyleId[] {
  const wanted = new Set(adapters);
  return STYLE_IDS.filter((style) => wanted.has(style));
}

export class ModelLibrary {
  private store: Promise<ModelStore> | null = null;
  private download: AbortController | null = null;
  private removing = false;
  private inspection: Promise<void> = Promise.resolve();

  constructor(
    private readonly sourceUrl: string,
    private readonly deps: LibraryDeps,
  ) {}

  /**
   * Report what is installed; called when the worker starts. A model this
   * build cannot use (from another source, or stored by an older version) is
   * deleted rather than left taking up disk space.
   */
  inspect(): Promise<void> {
    // Downloads and removals wait for it: its clean-up must not delete files they are writing.
    this.inspection = this.inspectStore();
    return this.inspection;
  }

  private async inspectStore(): Promise<void> {
    const store = await this.openStore();
    const model = await store.installed();
    const usable = model && model.sourceUrl === this.sourceUrl ? model : null;
    if (!usable && (await store.hasRecord())) await store.clear();
    this.deps.emit({ type: 'installed', installed: usable ? toRecord(usable) : null });
  }

  /** The installed model and its store, for the engine to load. */
  async require(): Promise<{ model: StoredModel; store: ModelStore }> {
    // A worker asked to load the model as it starts must not read the record while the inspection holds it open.
    await this.inspection.catch(() => undefined);
    const model = this.removing ? null : await this.current();
    if (!model) {
      // The files may have gone behind the mirror's back (the browser can evict site data).
      this.deps.emit({ type: 'installed', installed: null });
      throw new EchoError('MODEL_NOT_DOWNLOADED');
    }
    return { model, store: await this.openStore() };
  }

  /** Download `adapters`, and the base first if it is not installed yet. */
  async startDownload(adapters: readonly StyleId[]): Promise<void> {
    if (this.download || this.removing) return;
    const controller = new AbortController();
    this.download = controller;
    await this.inspection.catch(() => undefined);
    const emitDownload = (download: DownloadState) => this.deps.emit({ type: 'download', download });
    let target: DownloadTarget = { base: true, adapters: inStyleOrder(adapters) };
    try {
      const store = await this.openStore();
      const installed = await this.current();
      target = { base: installed === null, adapters: target.adapters };
      emitDownload({ state: 'downloading', target, phase: 'fetching', receivedBytes: 0, totalBytes: 0, bytesPerSecond: 0 });

      const manifest = await fetchManifest(this.sourceUrl, this.deps.transport, controller.signal);
      if (installed && baseFingerprint(installed.manifest) !== baseFingerprint(manifest)) throw new EchoError('MODEL_OUTDATED');
      const unknown = target.adapters.find((style) => !manifest.adapters[style]);
      if (unknown) throw new EchoError('DOWNLOAD_FAILED', { reason: `model.json has no ${unknown} adapter` });

      // Adapters already installed stay installed under the new manifest; any whose files changed are fetched again.
      const next: StoredModel = {
        sourceUrl: this.sourceUrl,
        manifest,
        adapters: inStyleOrder([...(installed?.adapters ?? []).filter((style) => manifest.adapters[style]), ...target.adapters]),
      };
      const files = filesToFetch(next, installed);
      await downloadFiles(store, this.sourceUrl, files, this.deps.transport, {
        signal: controller.signal,
        onProgress: (progress) => emitDownload({ state: 'downloading', target, ...progress }),
      });

      await store.commit(next);
      await store.prune(ModelStore.namesOf(next));
      this.deps.emit({ type: 'installed', installed: toRecord(next) });
      emitDownload({ state: 'idle' });
    } catch (thrown) {
      const error = downloadError(thrown, controller.signal);
      if (error.code === 'CANCELLED') {
        // Best effort: files left behind are reused or cleaned up by the next download.
        await this.discardPartialFiles().catch(() => undefined);
        emitDownload({ state: 'idle' });
      } else {
        // Partial files stay on disk: trying again resumes them.
        emitDownload({ state: 'failed', target, error: toErrorPayload(error, 'DOWNLOAD_FAILED') });
      }
    } finally {
      this.download = null;
    }
  }

  cancelDownload(): void {
    this.download?.abort();
  }

  /** Delete one adapter; the base and the other adapters stay. */
  async removeAdapter(style: StyleId): Promise<void> {
    if (this.download || this.removing) return;
    await this.inspection.catch(() => undefined);
    const installed = await this.current();
    if (installed?.adapters.includes(style)) {
      const next: StoredModel = { ...installed, adapters: installed.adapters.filter((adapter) => adapter !== style) };
      const store = await this.openStore();
      // Recorded first: a crash in between leaves stray files, never a record naming missing ones.
      await store.commit(next);
      await store.prune(ModelStore.namesOf(next));
    }
    await this.inspectStore();
  }

  /** Delete the base and every adapter; a rewrite in progress finishes first. */
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
    await store.prune(installed ? ModelStore.namesOf(installed) : new Set());
  }
}
