import { describe, expect, jest, test } from '@jest/globals';
import { parseModelManifest } from '../../../engine/manifest';
import type { EngineEvent } from '../../../shared/messages';
import { ModelLibrary } from '../../../worker/modelLibrary';
import { sha256Hex } from '../../../worker/modelDownloader';
import { ModelStore } from '../../../worker/modelStore';
import { readJsonFixture } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, modelServer, tinyModelFiles, type ServerOptions } from '../../helpers/modelServer';

const manifest = parseModelManifest(readJsonFixture('tiny-t5-int8/model.json'));

function setup(serverOptions: ServerOptions = {}, sourceUrl = FIXTURE_URL, root = new MemoryDirectory()) {
  const events: EngineEvent[] = [];
  /** Runs after each event is recorded; tests use it to act at a precise moment. */
  const hooks = { onEvent: (_event: EngineEvent) => undefined as void };
  const server = modelServer(tinyModelFiles(), serverOptions);
  const engine = { unload: jest.fn(async () => undefined) };
  const library = new ModelLibrary(sourceUrl, {
    openStore: () => ModelStore.open(root),
    transport: { fetch: server.fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 },
    engine,
    emit: (event) => {
      events.push(event);
      hooks.onEvent(event);
    },
  });
  const folder = () => root.getDirectoryHandle('model', { create: true });
  const installedEvents = () => events.filter((event) => event.type === 'installed');
  const downloadEvents = () => events.flatMap((event) => (event.type === 'download' ? [event.download] : []));
  return { root, library, events, hooks, server, engine, folder, installedEvents, downloadEvents };
}

describe('ModelLibrary', () => {
  test('reports nothing installed on a fresh profile', async () => {
    const { library, installedEvents } = setup();
    await library.inspect();
    expect(installedEvents()).toEqual([{ type: 'installed', installed: null }]);
    await expect(library.require()).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });

  test('a download reports progress, then the installed model, then idle', async () => {
    const { library, installedEvents, downloadEvents } = setup();
    await library.startDownload();

    const states = downloadEvents();
    expect(states[0]).toEqual({ state: 'downloading', phase: 'fetching', receivedBytes: 0, totalBytes: 0, bytesPerSecond: 0 });
    expect(states.some((state) => state.state === 'downloading' && state.phase === 'verifying')).toBe(true);
    expect(states.at(-1)).toEqual({ state: 'idle' });
    expect(installedEvents().at(-1)).toEqual({
      type: 'installed',
      installed: { sourceUrl: FIXTURE_URL, model: expect.objectContaining({ id: manifest.id, sizeBytes: expect.any(Number) }) },
    });
    const { model } = await library.require();
    expect(model.manifest).toEqual(manifest);
  });

  test('asking again while a download runs starts nothing new', async () => {
    const { library, server } = setup();
    await Promise.all([library.startDownload(), library.startDownload()]);
    expect(server.requests.filter((request) => request.url.endsWith('model.json'))).toHaveLength(1);
  });

  test('an installed model is not downloaded again', async () => {
    const { library, server, downloadEvents } = setup();
    await library.startDownload();
    const requests = server.requests.length;
    await library.startDownload();
    expect(server.requests.length).toBe(requests);
    expect(downloadEvents().at(-1)).toEqual({ state: 'idle' });
  });

  test('cancelling deletes the partly downloaded files', async () => {
    const { library, hooks, folder, downloadEvents } = setup({ chunkSize: 512 });
    hooks.onEvent = (event) => {
      if (event.type === 'download' && event.download.state === 'downloading' && event.download.receivedBytes > 0) library.cancelDownload();
    };
    await library.startDownload();
    expect(downloadEvents().some((state) => state.state === 'downloading' && state.receivedBytes > 0)).toBe(true);
    expect(downloadEvents().at(-1)).toEqual({ state: 'idle' });
    expect((await folder()).names()).toEqual([]);
  });

  test('a failed download is reported and keeps its partial files for the retry', async () => {
    const { library, folder, downloadEvents } = setup({ dropAfter: { path: 'encoder.onnx', bytes: 20_000 } });
    await library.startDownload();
    expect(downloadEvents().at(-1)).toMatchObject({ state: 'failed', error: { code: 'DOWNLOAD_FAILED' } });
    expect((await folder()).names()).toContain(manifest.files.encoder.sha256);
  });

  test('a model downloaded from another URL does not count as installed', async () => {
    const first = setup();
    await first.library.startDownload();
    // The same storage, read by a build whose model.source.json points somewhere else.
    const next = setup({}, 'https://models.example.test/next/', first.root);
    await next.library.inspect();
    expect(next.installedEvents()).toEqual([{ type: 'installed', installed: null }]);
  });

  test('removing unloads the engine first, deletes every file and reports nothing installed', async () => {
    const { library, engine, folder, installedEvents } = setup();
    await library.startDownload();
    await library.remove();
    expect(engine.unload).toHaveBeenCalledTimes(1);
    expect((await folder()).names()).toEqual([]);
    expect(installedEvents().at(-1)).toEqual({ type: 'installed', installed: null });
  });

  test('the engine cannot load the model while it is being removed', async () => {
    const { library, engine } = setup();
    await library.startDownload();
    let finishUnload!: () => void;
    engine.unload.mockImplementationOnce(() => new Promise<undefined>((resolve) => (finishUnload = () => resolve(undefined))));
    const removing = library.remove();
    await expect(library.require()).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
    finishUnload();
    await removing;
  });

  test('files that vanished behind the mirror are reported when the engine asks for them', async () => {
    const { library, folder, installedEvents } = setup();
    await library.startDownload();
    await (await folder()).removeEntry(manifest.files.decoder.sha256);
    await expect(library.require()).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
    expect(installedEvents().at(-1)).toEqual({ type: 'installed', installed: null });
  });
});
