/**
 * The service worker's half of downloading the model "in place": the card a
 * page shows when a rewrite needs the model (or a style) that is not on this
 * device yet.
 *
 * Each open card holds a port. The card says what it is missing; the broker
 * works out what the download would be (the model with the style asked for
 * and the starter style, or the one style), starts it when asked, and keeps every open card told how it
 * is going, from the events the engine reports. The download itself is the
 * same one the popup starts; closing a card never stops it.
 */
import { toErrorPayload } from '../shared/errors';
import type { EngineEvent, SetupEvent, SetupNeed, SetupRequest } from '../shared/messages';
import { isSetupRequest } from '../shared/messages';
import { adapterForStyle, IDLE_DOWNLOAD, modelFrom, offeredCatalog, starterAdapter, type DownloadState, type InstalledModelRecord, type ModelCatalog } from '../shared/modelInstall';
import type { StyleId } from '../shared/styles';

export interface SetupPortLike {
  postMessage(event: SetupEvent): void;
  onMessage: { addListener(listener: (message: unknown) => void): void };
  onDisconnect: { addListener(listener: () => void): void };
}

export interface SetupBrokerDeps {
  /** What the download server offers; rejects when it can't be reached. */
  readonly fetchCatalog: () => Promise<ModelCatalog>;
  /** The installed model and the download's state, as stored, for when the worker has just woken up. */
  readonly readInstalled: () => Promise<InstalledModelRecord | null>;
  readonly readDownload: () => Promise<DownloadState>;
  /** The folder this build downloads from; a model from another folder counts as not installed. */
  readonly sourceUrl: string;
  /** Start downloading these styles (and the base, if it is not installed). */
  readonly startDownload: (adapters: readonly StyleId[]) => void;
}

interface Plan {
  readonly adapters: StyleId[];
  readonly totalBytes: number;
  /** The style that has to be installed for the card's job to be done. */
  readonly requires: StyleId;
}

interface Session {
  readonly port: SetupPortLike;
  plan: Plan | null;
}

export class SetupBroker {
  private readonly sessions = new Set<Session>();
  private installed: InstalledModelRecord | null | undefined;
  private download: DownloadState = IDLE_DOWNLOAD;

  constructor(private readonly deps: SetupBrokerDeps) {}

  handlePort(port: SetupPortLike): void {
    const session: Session = { port, plan: null };
    this.sessions.add(session);
    port.onDisconnect.addListener(() => this.sessions.delete(session));
    port.onMessage.addListener((message) => {
      if (!isSetupRequest(message)) return;
      void this.handle(session, message);
    });
  }

  /** The engine's events, as they come: the broker keeps its own picture of the install and the download. */
  onEvent(event: EngineEvent): void {
    if (event.type === 'installed') this.installed = modelFrom(event.installed, this.deps.sourceUrl);
    else if (event.type === 'download') this.download = event.download;
    else return;
    for (const session of this.sessions) this.report(session);
  }

  private async handle(session: Session, request: SetupRequest): Promise<void> {
    try {
      if (request.kind === 'setup/download') {
        if (session.plan) this.deps.startDownload(session.plan.adapters);
        return;
      }
      await this.open(session, request.need);
    } catch (error) {
      this.send(session, { kind: 'setup/failed', error: toErrorPayload(error, 'DOWNLOAD_FAILED') });
    }
  }

  private async open(session: Session, need: SetupNeed): Promise<void> {
    if (this.installed === undefined) {
      const [installed, download] = await Promise.all([this.deps.readInstalled(), this.deps.readDownload()]);
      // An event that arrived while reading is newer than what was stored.
      if (this.installed === undefined) this.installed = modelFrom(installed, this.deps.sourceUrl);
      if (this.download === IDLE_DOWNLOAD) this.download = download;
    }
    const installed = this.installed ?? null;
    // The server's catalog fits the installed base, or else the installed one stands in for it (an unreachable server is no reason to refuse a style that is already listed).
    let remote: ModelCatalog | null = null;
    try {
      remote = await this.deps.fetchCatalog();
    } catch (error) {
      if (!installed) throw error;
    }
    const catalog = offeredCatalog(installed, remote);
    if (!catalog) throw new Error('no catalog');
    session.plan = planFor(need, catalog, installed);
    this.report(session);
  }

  /** Tell one card where things stand. */
  private report(session: Session): void {
    const { plan } = session;
    if (!plan) return;
    const { download } = this;
    if (download.state === 'downloading') {
      this.send(session, {
        kind: 'setup/progress',
        verifying: download.phase === 'verifying',
        receivedBytes: download.receivedBytes,
        totalBytes: download.totalBytes || plan.totalBytes,
        bytesPerSecond: download.bytesPerSecond,
      });
    } else if (download.state === 'failed') {
      this.send(session, { kind: 'setup/failed', error: download.error });
    } else if (this.installed?.adapters.includes(plan.requires)) {
      this.send(session, { kind: 'setup/done' });
    } else {
      this.send(session, { kind: 'setup/offer', totalBytes: plan.totalBytes, styles: plan.adapters });
    }
  }

  private send(session: Session, event: SetupEvent): void {
    try {
      session.port.postMessage(event);
    } catch {
      // The page went away; its disconnect removes the session.
      this.sessions.delete(session);
    }
  }
}

/**
 * What to download for `need`: the one style, or the base with the style asked for. The base always comes
 * with the starter style too (as it does from the popup), so the first download leaves the user with it.
 */
export function planFor(need: SetupNeed, catalog: ModelCatalog, installed: InstalledModelRecord | null): Plan {
  const starter = starterAdapter(catalog);
  const style: StyleId | undefined = need.kind === 'style' ? need.style : need.style ?? starter?.style;
  const adapter = style ? catalog.adapters.find((candidate) => candidate.style === (adapterForStyle(catalog, style) ?? style)) : undefined;
  if (!adapter) throw new Error(`model.json has no ${style ?? 'starter'} style`);
  const adapters = installed || !starter || starter.style === adapter.style ? [adapter] : [adapter, starter];
  const baseBytes = installed ? 0 : catalog.model.sizeBytes;
  return {
    adapters: adapters.map((candidate) => candidate.style),
    totalBytes: baseBytes + adapters.reduce((sum, candidate) => sum + candidate.sizeBytes, 0),
    requires: adapter.style,
  };
}
