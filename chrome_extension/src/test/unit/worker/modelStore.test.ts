import { describe, expect, test } from '@jest/globals';
import { parseModelManifest } from '../../../engine/manifest';
import { ModelStore } from '../../../worker/modelStore';
import { readJsonFixture } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { tinyModelFiles } from '../../helpers/modelServer';

const raw = readJsonFixture('tiny-t5-int8/model.json');
const manifest = parseModelManifest(raw);
const served = tinyModelFiles();
const SOURCE = 'https://models.example.test/tiny/';

async function storeWith(files: 'all' | 'none' | 'truncated-decoder' = 'all') {
  const root = new MemoryDirectory();
  const store = await ModelStore.open(root);
  const folder = await root.getDirectoryHandle('model');
  if (files !== 'none') {
    for (const file of Object.values(manifest.files)) {
      let bytes = served.get(file.path)!;
      if (files === 'truncated-decoder' && file === manifest.files.decoder) bytes = bytes.subarray(0, 100);
      const handle = await store.openForWrite(file);
      handle.write(bytes, { at: 0 });
      handle.close();
    }
  }
  return { store, folder };
}

describe('ModelStore', () => {
  test('nothing is installed until the record is committed', async () => {
    const { store } = await storeWith('all');
    expect(await store.installed()).toBeNull();
    await store.commit(SOURCE, raw);
    expect(await store.installed()).toEqual({ sourceUrl: SOURCE, manifest });
  });

  test('files are stored under their sha256, next to the record', async () => {
    const { store, folder } = await storeWith('all');
    await store.commit(SOURCE, raw);
    expect(folder.names()).toEqual([...ModelStore.namesOf(manifest)].sort());
    expect(folder.names()).toContain(manifest.files.encoder.sha256);
  });

  test('a record whose files are missing or short does not count as installed', async () => {
    const truncated = await storeWith('truncated-decoder');
    await truncated.store.commit(SOURCE, raw);
    expect(await truncated.store.installed()).toBeNull();

    const empty = await storeWith('none');
    await empty.store.commit(SOURCE, raw);
    expect(await empty.store.installed()).toBeNull();
  });

  test('an unreadable record does not count as installed', async () => {
    const { store, folder } = await storeWith('all');
    const record = await folder.getFileHandle('installed.json', { create: true });
    const handle = await record.createSyncAccessHandle();
    handle.write(new TextEncoder().encode('{"sourceUrl": "x", "manif'), { at: 0 });
    handle.close();
    expect(await store.installed()).toBeNull();
  });

  test('read returns the whole file, reporting progress, and refuses an incomplete one', async () => {
    const { store } = await storeWith('truncated-decoder');
    const seen: number[] = [];
    const encoder = await store.read(manifest.files.encoder, (count) => seen.push(count));
    expect(encoder).toEqual(served.get('encoder.onnx'));
    expect(seen.reduce((sum, count) => sum + count, 0)).toBe(manifest.files.encoder.bytes);
    await expect(store.read(manifest.files.decoder)).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });

  test('a missing file reads as size 0 and cannot be read', async () => {
    const { store } = await storeWith('none');
    expect(await store.size(manifest.files.encoder)).toBe(0);
    await expect(store.read(manifest.files.encoder)).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });

  test('prune keeps only the named entries; clear removes everything', async () => {
    const { store, folder } = await storeWith('all');
    await store.commit(SOURCE, raw);
    const stray = await folder.getFileHandle('f'.repeat(64), { create: true });
    (await stray.createSyncAccessHandle()).close();

    await store.prune(ModelStore.namesOf(manifest));
    expect(folder.names()).toEqual([...ModelStore.namesOf(manifest)].sort());
    await store.clear();
    expect(folder.names()).toEqual([]);
    await store.remove(manifest.files.encoder); // already gone: not an error
  });

  test('every access handle is closed again', async () => {
    const { store, folder } = await storeWith('all');
    await store.commit(SOURCE, raw);
    await store.installed();
    await store.read(manifest.files.tokenizer);
    for (const name of folder.names()) expect(folder.file(name).locked).toBe(false);
  });
});
