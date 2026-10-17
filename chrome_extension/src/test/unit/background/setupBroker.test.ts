import { describe, expect, jest, test } from '@jest/globals';
import { planFor, SetupBroker, type SetupBrokerDeps, type SetupPortLike } from '../../../background/setupBroker';
import type { SetupEvent, SetupRequest } from '../../../shared/messages';
import { IDLE_DOWNLOAD, type DownloadState, type InstalledModelRecord } from '../../../shared/modelInstall';
import type { StyleId } from '../../../shared/styles';
import { installedRecord, TINY_CATALOG } from '../../helpers/fixtures';

const SOURCE = 'http://localhost:47615/echomebetter/';
const BASE = TINY_CATALOG.model.sizeBytes;
const PROFESSIONAL = TINY_CATALOG.adapters.find((adapter) => adapter.style === 'professional')!.sizeBytes;

function setup(options: { installed?: InstalledModelRecord | null; download?: DownloadState; catalogFails?: boolean } = {}) {
  const startDownload = jest.fn((_adapters: readonly StyleId[]) => undefined);
  const deps: SetupBrokerDeps = {
    fetchCatalog: async () => {
      if (options.catalogFails) throw new Error('offline');
      return TINY_CATALOG;
    },
    readInstalled: async () => options.installed ?? null,
    readDownload: async () => options.download ?? IDLE_DOWNLOAD,
    sourceUrl: SOURCE,
    startDownload,
  };
  const broker = new SetupBroker(deps);
  const events: SetupEvent[] = [];
  let send!: (message: unknown) => void;
  let disconnect!: () => void;
  const port: SetupPortLike = {
    postMessage: (event) => void events.push(event),
    onMessage: { addListener: (listener) => (send = listener) },
    onDisconnect: { addListener: (listener) => (disconnect = listener) },
  };
  broker.handlePort(port);
  const ask = async (request: SetupRequest) => {
    send(request);
    await new Promise((resolve) => setTimeout(resolve, 0));
  };
  return { broker, events, ask, startDownload, disconnect: () => disconnect() };
}

describe('SetupBroker', () => {
  test('offers the model with the starter style, sized, and downloads that when asked', async () => {
    const { events, ask, startDownload } = setup();
    await ask({ kind: 'setup/open', need: { kind: 'model' } });
    expect(events).toEqual([{ kind: 'setup/offer', totalBytes: BASE + PROFESSIONAL, styles: ['professional'] }]);
    await ask({ kind: 'setup/download' });
    expect(startDownload).toHaveBeenCalledWith(['professional']);
  });

  test('a missing model asked for in a style brings the style that makes it work', async () => {
    const { events, ask, startDownload } = setup();
    await ask({ kind: 'setup/open', need: { kind: 'model', style: 'concise' } });
    expect(events).toEqual([{ kind: 'setup/offer', totalBytes: BASE + PROFESSIONAL, styles: ['professional'] }]);
    await ask({ kind: 'setup/download' });
    expect(startDownload).toHaveBeenCalledWith(['professional']);
  });

  test('a missing style alone is only the style', async () => {
    const { events, ask, startDownload } = setup({ installed: installedRecord(SOURCE, []) });
    await ask({ kind: 'setup/open', need: { kind: 'style', style: 'professional' } });
    expect(events).toEqual([{ kind: 'setup/offer', totalBytes: PROFESSIONAL, styles: ['professional'] }]);
    await ask({ kind: 'setup/download' });
    expect(startDownload).toHaveBeenCalledWith(['professional']);
  });

  test('follows the download to the end: progress, verifying, then done once the model is installed', async () => {
    const { broker, events, ask } = setup();
    await ask({ kind: 'setup/open', need: { kind: 'model' } });
    events.length = 0;
    const target = { base: true, adapters: ['professional' as const] };
    broker.onEvent({ type: 'download', download: { state: 'downloading', target, phase: 'fetching', receivedBytes: 100, totalBytes: 1000, bytesPerSecond: 50 } });
    broker.onEvent({ type: 'download', download: { state: 'downloading', target, phase: 'verifying', receivedBytes: 1000, totalBytes: 1000, bytesPerSecond: 0 } });
    // The worker reports the install before the download is idle again.
    broker.onEvent({ type: 'installed', installed: installedRecord(SOURCE) });
    broker.onEvent({ type: 'download', download: IDLE_DOWNLOAD });
    expect(events).toEqual([
      { kind: 'setup/progress', verifying: false, receivedBytes: 100, totalBytes: 1000, bytesPerSecond: 50 },
      { kind: 'setup/progress', verifying: true, receivedBytes: 1000, totalBytes: 1000, bytesPerSecond: 0 },
      { kind: 'setup/progress', verifying: true, receivedBytes: 1000, totalBytes: 1000, bytesPerSecond: 0 },
      { kind: 'setup/done' },
    ]);
  });

  test('a failed download is reported, and asking again starts it again', async () => {
    const { broker, events, ask, startDownload } = setup();
    await ask({ kind: 'setup/open', need: { kind: 'model' } });
    broker.onEvent({ type: 'download', download: { state: 'failed', target: { base: true, adapters: ['professional'] }, error: { code: 'DOWNLOAD_FAILED' } } });
    expect(events.at(-1)).toEqual({ kind: 'setup/failed', error: { code: 'DOWNLOAD_FAILED' } });
    await ask({ kind: 'setup/download' });
    expect(startDownload).toHaveBeenCalledTimes(1);
  });

  test('a download cancelled elsewhere (the popup) puts the offer back', async () => {
    const { broker, events, ask } = setup();
    await ask({ kind: 'setup/open', need: { kind: 'model' } });
    broker.onEvent({ type: 'download', download: IDLE_DOWNLOAD });
    expect(events.at(-1)).toEqual({ kind: 'setup/offer', totalBytes: BASE + PROFESSIONAL, styles: ['professional'] });
  });

  test('a card opened while a download is already running joins it; one opened when nothing is missing any more says so', async () => {
    const running = setup({ download: { state: 'downloading', target: { base: true, adapters: ['professional'] }, phase: 'fetching', receivedBytes: 5, totalBytes: 0, bytesPerSecond: 0 } });
    await running.ask({ kind: 'setup/open', need: { kind: 'model' } });
    expect(running.events).toEqual([{ kind: 'setup/progress', verifying: false, receivedBytes: 5, totalBytes: BASE + PROFESSIONAL, bytesPerSecond: 0 }]);

    const installed = setup({ installed: installedRecord(SOURCE) });
    await installed.ask({ kind: 'setup/open', need: { kind: 'model' } });
    expect(installed.events).toEqual([{ kind: 'setup/done' }]);
  });

  test("a model from another download folder doesn't count", async () => {
    const { events, ask } = setup({ installed: installedRecord('https://elsewhere.example/model/') });
    await ask({ kind: 'setup/open', need: { kind: 'model' } });
    expect(events).toEqual([{ kind: 'setup/offer', totalBytes: BASE + PROFESSIONAL, styles: ['professional'] }]);
  });

  test('an unreachable server is a failure to show, unless a model is already on the device', async () => {
    const offline = setup({ catalogFails: true });
    await offline.ask({ kind: 'setup/open', need: { kind: 'model' } });
    expect(offline.events).toEqual([{ kind: 'setup/failed', error: { code: 'DOWNLOAD_FAILED', details: { message: 'offline' } } }]);

    const known = setup({ catalogFails: true, installed: installedRecord(SOURCE, []) });
    await known.ask({ kind: 'setup/open', need: { kind: 'style', style: 'professional' } });
    expect(known.events).toEqual([{ kind: 'setup/offer', totalBytes: PROFESSIONAL, styles: ['professional'] }]);
  });

  test('ignores what it does not understand, and stops talking to a card that has closed', async () => {
    const { broker, events, ask, disconnect } = setup();
    await ask({ kind: 'nonsense' } as never);
    expect(events).toEqual([]);
    await ask({ kind: 'setup/open', need: { kind: 'model' } });
    disconnect();
    events.length = 0;
    broker.onEvent({ type: 'download', download: IDLE_DOWNLOAD });
    expect(events).toEqual([]);
  });
});

describe('planFor', () => {
  test('names the style that has to be on the device when it is done', () => {
    expect(planFor({ kind: 'model' }, TINY_CATALOG, null)).toMatchObject({ adapters: ['professional'], requires: 'professional' });
    expect(() => planFor({ kind: 'style', style: 'grammar' }, TINY_CATALOG, null)).not.toThrow();
  });

  const WITH_CONCISE = { ...TINY_CATALOG, adapters: [...TINY_CATALOG.adapters, { style: 'concise' as const, sizeBytes: 7 }] };

  test('the writing model always comes with the starter style, after the style asked for', () => {
    expect(planFor({ kind: 'model', style: 'concise' }, WITH_CONCISE, null)).toEqual({
      adapters: ['concise', 'professional'],
      totalBytes: BASE + 7 + PROFESSIONAL,
      requires: 'concise',
    });
  });

  test('once the model is installed, a style comes alone', () => {
    expect(planFor({ kind: 'style', style: 'concise' }, WITH_CONCISE, installedRecord(SOURCE))).toEqual({ adapters: ['concise'], totalBytes: 7, requires: 'concise' });
  });
});
