import { describe, expect, jest, test } from '@jest/globals';
import { remoteCatalogCache } from '../../../background/remoteCatalog';
import { newerStyleProblem, type InstalledModelRecord, type ModelCatalog } from '../../../shared/modelInstall';
import type { StyleId } from '../../../shared/styles';

const catalog = (adapters: StyleId[], base = 'base-1'): ModelCatalog => ({
  base,
  model: { id: 'm', displayName: 'EchoMeBetter', precision: 'int8', sizeBytes: 790e6 },
  adapters: adapters.map((style) => ({ style, sizeBytes: 73e6 })),
  fallbackAdapter: 'professional',
});
const installed = (adapters: StyleId[], on: ModelCatalog = catalog(['professional'])): InstalledModelRecord => ({ sourceUrl: 'x', catalog: on, adapters });

describe('newerStyleProblem', () => {
  const server = catalog(['professional', 'concise']);

  test('a style standing in on Professional is to be downloaded once the server has its own adapter', () => {
    expect(newerStyleProblem('concise', installed(['professional']), server)).toEqual({
      code: 'STYLE_NOT_DOWNLOADED',
      details: { style: 'Concise', adapter: 'Concise' },
    });
  });

  test('nothing to do when the server has no adapter for it, for another base, or when the device knows it already', () => {
    expect(newerStyleProblem('grammar', installed(['professional']), server)).toBeNull();
    expect(newerStyleProblem('concise', installed(['professional']), catalog(['professional', 'concise'], 'base-2'))).toBeNull();
    expect(newerStyleProblem('concise', installed(['professional']), null)).toBeNull();
    expect(newerStyleProblem('professional', installed(['professional']), server)).toBeNull();
    expect(newerStyleProblem('concise', installed(['professional', 'concise'], server), server)).toBeNull();
  });
});

describe('remoteCatalogCache', () => {
  test('reads the server once while fresh, and again after it has gone stale', async () => {
    let time = 0;
    const fetchCatalog = jest.fn(async () => catalog(['professional']));
    const read = remoteCatalogCache({ fetchCatalog, now: () => time });
    await Promise.all([read(), read()]);
    await read();
    expect(fetchCatalog).toHaveBeenCalledTimes(1);
    time = 11 * 60_000;
    await read();
    expect(fetchCatalog).toHaveBeenCalledTimes(2);
  });

  test('an unreachable server is no catalog, and is tried again a minute later', async () => {
    let time = 0;
    const fetchCatalog = jest.fn(async (): Promise<ModelCatalog> => {
      throw new Error('offline');
    });
    const read = remoteCatalogCache({ fetchCatalog, now: () => time });
    expect(await read()).toBeNull();
    expect(await read()).toBeNull();
    expect(fetchCatalog).toHaveBeenCalledTimes(1);
    time = 61_000;
    await read();
    expect(fetchCatalog).toHaveBeenCalledTimes(2);
  });
});
