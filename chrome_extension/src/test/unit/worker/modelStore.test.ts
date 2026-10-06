import { describe, expect, test } from '@jest/globals';
import { adapterFiles, baseFiles, parseModelManifest, type ModelFile } from '../../../engine/manifest';
import { ModelStore, type StoredModel } from '../../../worker/modelStore';
import { readJsonFixture } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { tinyModelFiles } from '../../helpers/modelServer';

const manifest = parseModelManifest(readJsonFixture('tiny-echo-int8/model.json'));
const served = tinyModelFiles();
const SOURCE = 'https://models.example.test/tiny/';
const professional = adapterFiles(manifest.adapters.professional!);
const grammar = adapterFiles(manifest.adapters.grammar!);
const model = (adapters: StoredModel['adapters'] = ['professional']): StoredModel => ({ sourceUrl: SOURCE, manifest, adapters });

async function storeWith(files: readonly ModelFile[], truncate?: ModelFile) {
  const root = new MemoryDirectory();
  const store = await ModelStore.open(root);
  const folder = await root.getDirectoryHandle('model');
  for (const file of files) {
    let bytes = served.get(file.path)!;
    if (file === truncate) bytes = bytes.subarray(0, 100);
    const handle = await store.openForWrite(file);
    handle.write(bytes, { at: 0 });
    handle.close();
  }
  return { store, folder };
}

describe('ModelStore', () => {
  test('nothing is installed until the record is committed', async () => {
    const { store } = await storeWith([...baseFiles(manifest), ...professional]);
    expect(await store.installed()).toBeNull();
    expect(await store.hasRecord()).toBe(false);
    await store.commit(model());
    expect(await store.installed()).toEqual(model());
  });

  test('files are stored under their sha256, next to the record', async () => {
    const { store, folder } = await storeWith([...baseFiles(manifest), ...professional]);
    await store.commit(model());
    expect(folder.names()).toEqual([...ModelStore.namesOf(model())].sort());
    expect(folder.names()).toContain(manifest.files.encoder.sha256);
    expect(folder.names()).toContain(manifest.adapters.professional!.files.decoder.sha256);
  });

  test('a record whose base files are missing or short does not count as installed', async () => {
    const truncated = await storeWith(baseFiles(manifest), manifest.files.decoder);
    await truncated.store.commit(model([]));
    expect(await truncated.store.installed()).toBeNull();

    const empty = await storeWith([]);
    await empty.store.commit(model([]));
    expect(await empty.store.installed()).toBeNull();
    expect(await empty.store.hasRecord()).toBe(true);
  });

  test('an adapter whose files are not all there is left out; the base and the other adapters stay installed', async () => {
    const { store } = await storeWith([...baseFiles(manifest), ...professional, grammar[0]!]);
    await store.commit(model(['professional', 'grammar']));
    expect(await store.installed()).toEqual(model(['professional']));
  });

  test('an unreadable record does not count as installed', async () => {
    const { store, folder } = await storeWith(baseFiles(manifest));
    const record = await folder.getFileHandle('installed.json', { create: true });
    const handle = await record.createSyncAccessHandle();
    handle.write(new TextEncoder().encode('{"sourceUrl": "x", "manif'), { at: 0 });
    handle.close();
    expect(await store.installed()).toBeNull();
    expect(await store.hasRecord()).toBe(true);
  });

  test('read returns the whole file, reporting progress, and refuses an incomplete one', async () => {
    const { store } = await storeWith(baseFiles(manifest), manifest.files.decoder);
    const seen: number[] = [];
    const encoder = await store.read(manifest.files.encoder, (count) => seen.push(count));
    expect(encoder).toEqual(served.get('base/encoder.onnx'));
    expect(seen.reduce((sum, count) => sum + count, 0)).toBe(manifest.files.encoder.bytes);
    await expect(store.read(manifest.files.decoder)).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });

  test('a missing file reads as size 0 and cannot be read', async () => {
    const { store } = await storeWith([]);
    expect(await store.size(manifest.files.encoder)).toBe(0);
    await expect(store.read(manifest.files.encoder)).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });

  test('prune keeps only the named entries; clear removes everything', async () => {
    const { store, folder } = await storeWith([...baseFiles(manifest), ...professional, ...grammar]);
    await store.commit(model());
    await store.prune(ModelStore.namesOf(model()));
    expect(folder.names()).toEqual([...ModelStore.namesOf(model())].sort());
    expect(folder.names()).not.toContain(grammar[0]!.sha256);
    await store.clear();
    expect(folder.names()).toEqual([]);
    await store.remove(manifest.files.encoder); // already gone: not an error
  });

  test('every access handle is closed again', async () => {
    const { store, folder } = await storeWith([...baseFiles(manifest), ...professional]);
    await store.commit(model());
    await store.installed();
    await store.read(manifest.files.tokenizer);
    for (const name of folder.names()) expect(folder.file(name).locked).toBe(false);
  });
});
