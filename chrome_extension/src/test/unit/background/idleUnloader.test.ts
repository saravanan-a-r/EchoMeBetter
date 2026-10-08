import { expect, test } from '@jest/globals';
import { endRest, recordRest, shouldUnload, wakingFromRest } from '../../../background/idleUnloader';

const MINUTE = 60_000;

test.each([
  ['idle past the limit', { now: 20 * MINUTE, lastActivityAt: 0, activeJobs: 0, downloading: false, minutes: 15 }, true],
  ['idle but within the limit', { now: 10 * MINUTE, lastActivityAt: 0, activeJobs: 0, downloading: false, minutes: 15 }, false],
  ['a job is still running', { now: 99 * MINUTE, lastActivityAt: 0, activeJobs: 1, downloading: false, minutes: 5 }, false],
  ['the model is still downloading', { now: 99 * MINUTE, lastActivityAt: 0, activeJobs: 0, downloading: true, minutes: 5 }, false],
  ['"never" keeps the model loaded', { now: 999 * MINUTE, lastActivityAt: 0, activeJobs: 0, downloading: false, minutes: 0 }, false],
] as const)('%s', (_label, { now, lastActivityAt, activeJobs, downloading, minutes }, expected) => {
  expect(shouldUnload({ now, lastActivityAt, activeJobs, downloading, settings: { keepModelLoadedMinutes: minutes } })).toBe(expected);
});

test('a rewrite after the idle timer freed the model wakes it from a rest, until the model is loaded again', async () => {
  const data = new Map<string, unknown>();
  const storage = {
    get: async (key: string) => (data.has(key) ? { [key]: data.get(key) } : {}),
    set: async (items: Record<string, unknown>) => void Object.entries(items).forEach(([key, value]) => data.set(key, value)),
    remove: async (key: string) => void data.delete(key),
  } as unknown as chrome.storage.StorageArea;

  expect(await wakingFromRest({ keepModelLoadedMinutes: 15 }, storage)).toBeNull();
  await recordRest(storage);
  expect(await wakingFromRest({ keepModelLoadedMinutes: 15 }, storage)).toEqual({ idleMinutes: 15 });
  // "Never" can't be lengthened: nothing to suggest.
  expect(await wakingFromRest({ keepModelLoadedMinutes: 0 }, storage)).toBeNull();
  await endRest(storage);
  expect(await wakingFromRest({ keepModelLoadedMinutes: 15 }, storage)).toBeNull();
});
