import { describe, expect, test } from '@jest/globals';
import { parseModelManifest } from '../../../engine/manifest';
import { downloadModel, sha256Hex, type DownloadProgress, type DownloadTransport } from '../../../worker/modelDownloader';
import { ModelStore } from '../../../worker/modelStore';
import { readJsonFixture } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, modelServer, tinyModelFiles, type ServerOptions } from '../../helpers/modelServer';

const manifest = parseModelManifest(readJsonFixture('tiny-t5-int8/model.json'));
const TOTAL = manifest.files.encoder.bytes + manifest.files.decoder.bytes + manifest.files.tokenizer.bytes;

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
  const run = (url = FIXTURE_URL) =>
    downloadModel(store, url, transport, { signal: controller.signal, onProgress: (update) => progress.push(update) });
  return { store, folder, server, progress, controller, run };
}

describe('downloadModel', () => {
  test('downloads every file, verifies it, and records the model last', async () => {
    const { store, folder, progress, run } = await setup();
    const model = await run();

    expect(model).toEqual({ sourceUrl: FIXTURE_URL, manifest });
    expect(await store.installed()).toEqual(model);
    expect(folder.file(manifest.files.decoder.sha256).data).toEqual(tinyModelFiles().get('decoder.onnx'));
    expect(folder.names()).toEqual([...ModelStore.namesOf(manifest)].sort());
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
    expect(server.requests.length).toBe(4);
    expect(server.requests.every((request) => request.cache === 'no-store')).toBe(true);
  });

  test('resumes a partly downloaded file with a range request', async () => {
    const { store, server, progress, run } = await setup();
    const encoder = manifest.files.encoder;
    const partial = await store.openForWrite(encoder);
    partial.write(tinyModelFiles().get('encoder.onnx')!.subarray(0, 50_000), { at: 0 });
    partial.close();

    await run();
    expect(server.requests.find((request) => request.url.endsWith('encoder.onnx'))!.range).toBe('bytes=50000-');
    expect(await store.read(encoder)).toEqual(tinyModelFiles().get('encoder.onnx'));
    expect(progress[0]!.receivedBytes).toBeGreaterThanOrEqual(50_000);
  });

  test('a server that ignores the range gets the file restarted, not corrupted', async () => {
    const { store, run } = await setup({ ignoreRange: true });
    const partial = await store.openForWrite(manifest.files.decoder);
    partial.write(tinyModelFiles().get('decoder.onnx')!.subarray(0, 10_000), { at: 0 });
    partial.close();

    await run();
    expect(await store.read(manifest.files.decoder)).toEqual(tinyModelFiles().get('decoder.onnx'));
  });

  test('a dropped connection fails the download but keeps what arrived, so the next try resumes', async () => {
    const files = tinyModelFiles();
    const first = await setup({ dropAfter: { path: 'decoder.onnx', bytes: 30_000 } }, files);
    await expect(first.run()).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED' });
    expect(await first.store.installed()).toBeNull();
    expect(await first.store.size(manifest.files.decoder)).toBe(30_000);

    // Same storage, healthy network.
    const healthy = modelServer(files);
    await downloadModel(
      first.store,
      FIXTURE_URL,
      { fetch: healthy.fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 },
      { signal: new AbortController().signal, onProgress: () => undefined },
    );
    expect(healthy.requests.find((request) => request.url.endsWith('decoder.onnx'))!.range).toBe('bytes=30000-');
    expect(healthy.requests.some((request) => request.url.endsWith('encoder.onnx'))).toBe(false);
    expect(await first.store.installed()).not.toBeNull();
  });

  test('a file that does not match its sha256 is deleted and nothing is installed', async () => {
    const files = tinyModelFiles();
    const tampered = new Uint8Array(files.get('decoder.onnx')!);
    tampered[1000] = tampered[1000]! ^ 0xff;
    files.set('decoder.onnx', tampered);
    const { store, run } = await setup({}, files);

    await expect(run()).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED', details: { reason: 'checksum mismatch', file: 'decoder.onnx' } });
    expect(await store.size(manifest.files.decoder)).toBe(0);
    expect(await store.installed()).toBeNull();
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
    expect(server.requests.map((request) => request.url)).toEqual([`${FIXTURE_URL}model.json`]);
    expect(folder.names()).toEqual([]);
  });

  test('a missing folder, a broken model.json and an unreachable host are download failures', async () => {
    const withoutManifest = tinyModelFiles();
    withoutManifest.delete('model.json');
    const missing = await setup({}, withoutManifest);
    await expect(missing.run()).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED', details: { reason: 'HTTP 404' } });

    const files = tinyModelFiles();
    files.set('model.json', new TextEncoder().encode(JSON.stringify({ schemaVersion: 99 })));
    const broken = await setup({}, files);
    await expect(broken.run()).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED', details: { reason: expect.stringContaining('schemaVersion') } });

    const offline = await setup();
    await expect(offline.run('https://offline.example.test/')).rejects.toMatchObject({ code: 'DOWNLOAD_FAILED', details: { reason: 'Failed to fetch' } });
  });

  test('cancelling stops the download', async () => {
    const { store, controller } = await setup({ chunkSize: 1024 });
    const server = modelServer(tinyModelFiles(), { chunkSize: 1024 });
    const pending = downloadModel(
      store,
      FIXTURE_URL,
      { fetch: server.fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 },
      {
        signal: controller.signal,
        // Abort the moment the first bytes are on disk.
        onProgress: (update) => {
          if (update.receivedBytes > 0) controller.abort();
        },
      },
    );
    await expect(pending).rejects.toMatchObject({ code: 'CANCELLED' });
    expect(await store.installed()).toBeNull();
  });

  test('a completed download removes the files of the model it replaces', async () => {
    const { store, folder, run } = await setup();
    const old = await store.openForWrite({ path: 'old.onnx', bytes: 1, sha256: 'e'.repeat(64) });
    old.write(new Uint8Array([1]), { at: 0 });
    old.close();
    await run();
    expect(folder.names()).not.toContain('e'.repeat(64));
  });
});
