import { describe, expect, jest, test } from '@jest/globals';
import { adapterFiles, baseFiles, catalogOf, parseModelManifest } from '../../../engine/manifest';
import type { EngineEvent } from '../../../shared/messages';
import { ModelLibrary } from '../../../worker/modelLibrary';
import { sha256Hex } from '../../../worker/modelDownloader';
import { ModelStore } from '../../../worker/modelStore';
import { readJsonFixture } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, modelServer, tinyModelFiles, type ServerOptions } from '../../helpers/modelServer';

const raw = readJsonFixture<Record<string, unknown>>('tiny-echo-int8/model.json');
const manifest = parseModelManifest(raw);
const names = (files: { sha256: string }[]) => files.map((file) => file.sha256);
const BASE = names(baseFiles(manifest));
const PROFESSIONAL = names(adapterFiles(manifest.adapters.professional!));
const GRAMMAR = names(adapterFiles(manifest.adapters.grammar!));

function setup(serverOptions: ServerOptions = {}, sourceUrl = FIXTURE_URL, root = new MemoryDirectory(), files = tinyModelFiles()) {
  const events: EngineEvent[] = [];
  /** Runs after each event is recorded; tests use it to act at a precise moment. */
  const hooks = { onEvent: (_event: EngineEvent) => undefined as void };
  const server = modelServer(files, serverOptions);
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
  const installedEvents = () => events.flatMap((event) => (event.type === 'installed' ? [event.installed] : []));
  const downloadEvents = () => events.flatMap((event) => (event.type === 'download' ? [event.download] : []));
  const fileRequests = () => server.requests.filter((request) => !request.url.endsWith('model.json')).map((request) => request.url.slice(FIXTURE_URL.length));
  return { root, library, events, hooks, server, engine, folder, installedEvents, downloadEvents, fileRequests };
}

describe('ModelLibrary', () => {
  test('reports nothing installed on a fresh profile', async () => {
    const { library, installedEvents } = setup();
    await library.inspect();
    expect(installedEvents()).toEqual([null]);
    await expect(library.require()).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });

  test('the first download brings the base with the chosen adapters, reporting progress, then the installed model', async () => {
    const { library, installedEvents, downloadEvents, folder } = setup();
    await library.startDownload(['professional']);

    const target = { base: true, adapters: ['professional'] };
    const states = downloadEvents();
    expect(states[0]).toEqual({ state: 'downloading', target, phase: 'fetching', receivedBytes: 0, totalBytes: 0, bytesPerSecond: 0 });
    expect(states.some((state) => state.state === 'downloading' && state.phase === 'verifying')).toBe(true);
    expect(states.at(-1)).toEqual({ state: 'idle' });
    expect(installedEvents().at(-1)).toEqual({ sourceUrl: FIXTURE_URL, catalog: catalogOf(manifest), adapters: ['professional'] });
    expect((await folder()).names()).toEqual(['installed.json', ...BASE, ...PROFESSIONAL].sort());
    const { model } = await library.require();
    expect(model.manifest).toEqual(manifest);
  });

  test('the base can come alone, and adapters can be added to it later without fetching it again', async () => {
    const { library, installedEvents, downloadEvents, fileRequests, server } = setup();
    await library.startDownload([]);
    expect(installedEvents().at(-1)).toMatchObject({ adapters: [] });

    server.requests.length = 0;
    await library.startDownload(['grammar', 'professional']);
    expect(downloadEvents().at(-2)).toMatchObject({ target: { base: false, adapters: ['professional', 'grammar'] } });
    expect(fileRequests().sort()).toEqual([
      'adapters/grammar/decoder.onnx_adapter',
      'adapters/grammar/encoder.onnx_adapter',
      'adapters/professional/decoder.onnx_adapter',
      'adapters/professional/encoder.onnx_adapter',
    ]);
    expect(installedEvents().at(-1)).toMatchObject({ adapters: ['professional', 'grammar'] });
  });

  test('asking again while a download runs starts nothing new', async () => {
    const { library, server } = setup();
    await Promise.all([library.startDownload(['professional']), library.startDownload(['grammar'])]);
    expect(server.requests.filter((request) => request.url.endsWith('model.json'))).toHaveLength(1);
  });

  test('what is installed is not downloaded again', async () => {
    const { library, server, downloadEvents, fileRequests } = setup();
    await library.startDownload(['professional']);
    server.requests.length = 0;
    await library.startDownload(['professional']);
    expect(fileRequests()).toEqual([]);
    expect(downloadEvents().at(-1)).toEqual({ state: 'idle' });
  });

  test('cancelling deletes the partly downloaded files and keeps what was installed', async () => {
    const { library, hooks, folder, downloadEvents, installedEvents } = setup({ chunkSize: 512 });
    await library.startDownload(['professional']);
    hooks.onEvent = (event) => {
      if (event.type === 'download' && event.download.state === 'downloading' && event.download.receivedBytes > 0) library.cancelDownload();
    };
    await library.startDownload(['grammar']);
    expect(downloadEvents().at(-1)).toEqual({ state: 'idle' });
    expect((await folder()).names()).toEqual(['installed.json', ...BASE, ...PROFESSIONAL].sort());
    expect(installedEvents().at(-1)).toMatchObject({ adapters: ['professional'] });
  });

  test('a failed download is reported with what it was for, and keeps its partial files for the retry', async () => {
    const { library, folder, downloadEvents } = setup({ dropAfter: { path: 'base/encoder.onnx', bytes: 20_000 } });
    await library.startDownload(['professional']);
    expect(downloadEvents().at(-1)).toMatchObject({ state: 'failed', target: { base: true, adapters: ['professional'] }, error: { code: 'DOWNLOAD_FAILED' } });
    expect((await folder()).names()).toContain(manifest.files.encoder.sha256);
  });

  test('an adapter the server does not have fails the download', async () => {
    const { library, downloadEvents } = setup();
    await library.startDownload(['elaborate']);
    expect(downloadEvents().at(-1)).toMatchObject({ state: 'failed', error: { code: 'DOWNLOAD_FAILED', details: { reason: 'model.json has no elaborate adapter' } } });
  });

  test('adapters for a different base are refused, and the installed model is left alone', async () => {
    const first = setup();
    await first.library.startDownload(['professional']);
    // The server now holds a newer base model.
    const files = tinyModelFiles();
    const newer = { ...raw, files: { ...(raw.files as object), tokenizer: { ...manifest.files.tokenizer, sha256: 'c'.repeat(64) } } };
    files.set('model.json', new TextEncoder().encode(JSON.stringify(newer)));
    const next = setup({}, FIXTURE_URL, first.root, files);
    await next.library.startDownload(['grammar']);
    expect(next.downloadEvents().at(-1)).toMatchObject({ state: 'failed', error: { code: 'MODEL_OUTDATED' } });
    expect((await next.library.require()).model.adapters).toEqual(['professional']);
  });

  test('a model downloaded from another URL is not installed for this build, and its files are deleted', async () => {
    const first = setup();
    await first.library.startDownload(['professional']);
    // The same storage, read by a build whose model.source.json points somewhere else.
    const next = setup({}, 'https://models.example.test/next/', first.root);
    await next.library.inspect();
    expect(next.installedEvents()).toEqual([null]);
    expect((await next.folder()).names()).toEqual([]);
  });

  test('a record left by an older version is deleted with its files', async () => {
    const { root, library, folder, installedEvents } = setup();
    const store = await ModelStore.open(root);
    const record = await (await folder()).getFileHandle('installed.json', { create: true });
    const handle = await record.createSyncAccessHandle();
    handle.write(new TextEncoder().encode(JSON.stringify({ sourceUrl: FIXTURE_URL, manifest: { schemaVersion: 1 } })), { at: 0 });
    handle.close();
    const stale = await store.openForWrite({ path: 'encoder.onnx', bytes: 1, sha256: 'e'.repeat(64) });
    stale.write(new Uint8Array([1]), { at: 0 });
    stale.close();

    await library.inspect();
    expect(installedEvents()).toEqual([null]);
    expect((await folder()).names()).toEqual([]);
  });

  test('removing an adapter deletes only its files; the base stays loaded', async () => {
    const { library, engine, folder, installedEvents } = setup();
    await library.startDownload(['professional', 'grammar']);
    await library.removeAdapter('grammar');
    expect(engine.unload).not.toHaveBeenCalled();
    expect((await folder()).names()).toEqual(['installed.json', ...BASE, ...PROFESSIONAL].sort());
    expect((await folder()).names()).not.toEqual(expect.arrayContaining(GRAMMAR));
    expect(installedEvents().at(-1)).toMatchObject({ adapters: ['professional'] });
  });

  test('removing the model unloads the engine first, deletes every file and reports nothing installed', async () => {
    const { library, engine, folder, installedEvents } = setup();
    await library.startDownload(['professional']);
    await library.remove();
    expect(engine.unload).toHaveBeenCalledTimes(1);
    expect((await folder()).names()).toEqual([]);
    expect(installedEvents().at(-1)).toBeNull();
  });

  test('the engine cannot load the model while it is being removed', async () => {
    const { library, engine } = setup();
    await library.startDownload(['professional']);
    let finishUnload!: () => void;
    engine.unload.mockImplementationOnce(() => new Promise<undefined>((resolve) => (finishUnload = () => resolve(undefined))));
    const removing = library.remove();
    await expect(library.require()).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
    finishUnload();
    await removing;
  });

  test('base files that vanished behind the mirror are reported when the engine asks for them', async () => {
    const { library, folder, installedEvents } = setup();
    await library.startDownload(['professional']);
    await (await folder()).removeEntry(manifest.files.decoder.sha256);
    await expect(library.require()).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
    expect(installedEvents().at(-1)).toBeNull();
  });
});
