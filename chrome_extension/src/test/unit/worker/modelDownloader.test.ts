import { describe, expect, test } from '@jest/globals';
import { adapterFiles, baseFiles, bytesOf, parseModelManifest, type ModelFile } from '../../../engine/manifest';
import { downloadFiles, fetchManifest, sha256Hex, type DownloadProgress, type DownloadTransport } from '../../../worker/modelDownloader';
import { ModelStore } from '../../../worker/modelStore';
import { readJsonFixture } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, modelServer, tinyModelFiles, type ServerOptions } from '../../helpers/modelServer';

const manifest = parseModelManifest(readJsonFixture('tiny-echo-int8/model.json'));
/** The base and the professional adapter: what a first download fetches. */
const FILES = [...baseFiles(manifest), ...adapterFiles(manifest.adapters.professional!)];
const TOTAL = bytesOf(FILES);
const signal = () => new AbortController().signal;

async function setup(serverOptions: ServerOptions = {}, files = tinyModelFiles(), space = { quota: 10e9, usage: 0 }) {
  const root = new MemoryDirectory();
  const store = await ModelStore.open(root);
  const folder = await root.getDirectoryHandle('model');
  const server = modelServer(files, serverOptions);
  let clock = 0;
  const transport: DownloadTransport = {
    fetch: server.fetch,
    estimate: async () => space,
    digest: sha256Hex,
    now: () => (clock += 100),
  };
  const progress: DownloadProgress[] = [];
  const controller = new AbortController();
  const run = (url = FIXTURE_URL, files: readonly ModelFile[] = FILES) =>
    downloadFiles(store, url, files, transport, { signal: controller.signal, onProgress: (update) => progress.push(update) });
  return { store, folder, server, progress, controller, run };
}

describe('fetchManifest', () => {
  test('reads and validates model.json', async () => {
    const { server } = await setup();
    const transport = { fetch: server.fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 };
    expect(await fetchManifest(FIXTURE_URL, transport, signal())).toEqual(manifest);
    expect(server.requests.map((request) => [request.url, request.cache])).toEqual([[`${FIXTURE_URL}model.json`, 'no-store']]);
  });
});

describe('downloadFiles', () => {
  test('downloads every file into its sha256 name and verifies it', async () => {
    const { store, folder, progress, run } = await setup();
    await run();

    expect(folder.file(manifest.files.decoder.sha256).data).toEqual(tinyModelFiles().get('base/decoder.onnx'));
    expect(folder.file(manifest.adapters.professional!.files.encoder.sha256).data).toEqual(tinyModelFiles().get('adapters/professional/encoder.onnx_adapter'));
    expect(folder.names()).toEqual(FILES.map((file) => file.sha256).sort());
    expect(await store.installed()).toBeNull(); // recording the model is the library's part
    for (const name of folder.names()) expect(folder.file(name).locked).toBe(false);

    const fetching = progress.filter((update) => update.phase === 'fetching');
    expect(fetching.length).toBeGreaterThan(0);
    expect(fetching.every((update) => update.totalBytes === TOTAL)).toBe(true);
    expect(fetching.at(-1)!.bytesPerSecond).toBeGreaterThan(0);
    expect(progress.at(-1)).toMatchObject({ phase: 'verifying', receivedBytes: TOTAL });
  });

  test('requests bypass the HTTP cache, so Chrome keeps no second copy', async () => {
    const { server, run } = await setup();
    await run();
    expect(server.requests.length).toBe(FILES.length);
    expect(server.requests.every((request) => request.cache === 'no-store')).toBe(true);
  });

  test('resumes a partly downloaded file with a range request', async () => {
    const { store, server, progress, run } = await setup();
    const encoder = manifest.files.encoder;
    const partial = await store.openForWrite(encoder);
    partial.write(tinyModelFiles().get('base/encoder.onnx')!.subarray(0, 50_000), { at: 0 });
    partial.close();

    await run();
    expect(server.requests.find((request) => request.url.endsWith('base/encoder.onnx'))!.range).toBe('bytes=50000-');
    expect(await store.read(encoder)).toEqual(tinyModelFiles().get('base/encoder.onnx'));
    expect(progress[0]!.receivedBytes).toBeGreaterThanOrEqual(50_000);
  });

  test('a server that ignores the range gets the file restarted, not corrupted', async () => {
    const { store, run } = await setup({ ignoreRange: true });
    const partial = await store.openForWrite(manifest.files.decoder);
    partial.write(tinyModelFiles().get('base/decoder.onnx')!.subarray(0, 10_000), { at: 0 });
    partial.close();

    await run();
    expect(await store.read(manifest.files.decoder)).toEqual(tinyModelFiles().get('base/decoder.onnx'));
  });

  test('a dropped connection fails the download but keeps what arrived, so the next try resumes', async () => {
    const files = tinyModelFiles();
    const first = await setup({ dropAfter: { path: 'base/decoder.onnx', bytes: 30_000 } }, files);
    await expect(first.run()).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED' });
    expect(await first.store.size(manifest.files.decoder)).toBe(30_000);

    // Same storage, healthy network.
    const healthy = modelServer(files);
    await downloadFiles(
      first.store,
      FIXTURE_URL,
      FILES,
      { fetch: healthy.fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 },
      { signal: signal(), onProgress: () => undefined },
    );
    expect(healthy.requests.find((request) => request.url.endsWith('base/decoder.onnx'))!.range).toBe('bytes=30000-');
    expect(healthy.requests.some((request) => request.url.endsWith('base/encoder.onnx'))).toBe(false);
    expect(await first.store.complete(FILES)).toBe(true);
  });

  test('a file that does not match its sha256 is deleted and nothing is installed', async () => {
    const files = tinyModelFiles();
    const tampered = new Uint8Array(files.get('adapters/professional/decoder.onnx_adapter')!);
    tampered[1000] = tampered[1000]! ^ 0xff;
    files.set('adapters/professional/decoder.onnx_adapter', tampered);
    const { store, run } = await setup({}, files);

    await expect(run()).rejects.toMatchObject({
      code: 'DOWNLOAD_FAILED',
      details: { reason: 'checksum mismatch', file: 'adapters/professional/decoder.onnx_adapter' },
    });
    expect(await store.size(manifest.adapters.professional!.files.decoder)).toBe(0);
  });

  test('a file longer than model.json declares is rejected', async () => {
    const files = tinyModelFiles();
    const longer = new Uint8Array(files.get('tokenizer.json')!.byteLength + 10);
    longer.set(files.get('tokenizer.json')!);
    files.set('tokenizer.json', longer);
    const { run } = await setup({}, files);
    await expect(run()).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED', details: { reason: 'file is larger than model.json declares' } });
  });

  test('not enough disk space fails before anything is fetched', async () => {
    const { server, folder, run } = await setup({}, tinyModelFiles(), { quota: 1_000_000, usage: 999_000 });
    await expect(run()).rejects.toMatchObject({ code: 'STORAGE_FULL', details: { neededBytes: expect.any(Number) } });
    expect(server.requests).toEqual([]);
    expect(folder.names()).toEqual([]);
  });

  test('a missing folder, a broken model.json and an unreachable host fail', async () => {
    const withoutManifest = tinyModelFiles();
    withoutManifest.delete('model.json');
    const missing = await setup({}, withoutManifest);
    const transport = (fetch: DownloadTransport['fetch']) => ({ fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 });
    await expect(fetchManifest(FIXTURE_URL, transport(missing.server.fetch), signal())).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED', details: { reason: 'HTTP 404' } });

    const files = tinyModelFiles();
    files.set('model.json', new TextEncoder().encode(JSON.stringify({ schemaVersion: 99 })));
    const broken = await setup({}, files);
    await expect(fetchManifest(FIXTURE_URL, transport(broken.server.fetch), signal())).rejects.toMatchObject({
      code: 'DOWNLOAD_FAILED',
      details: { reason: expect.stringContaining('schemaVersion') },
    });

    const offline = await setup();
    await expect(offline.run('https://offline.example.test/')).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED', details: { reason: 'Failed to fetch' } });
  });

  test('cancelling stops the download', async () => {
    const { store, controller } = await setup({ chunkSize: 1024 });
    const server = modelServer(tinyModelFiles(), { chunkSize: 1024 });
    const pending = downloadFiles(store, FIXTURE_URL, FILES, { fetch: server.fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 }, {
      signal: controller.signal,
      // Abort the moment the first bytes are on disk.
      onProgress: (update) => {
        if (update.receivedBytes > 0) controller.abort();
      },
    });
    await expect(pending).rejects.toMatchObject({ code: 'CANCELLED' });
    expect(await store.complete(FILES)).toBe(false);
  });
});
