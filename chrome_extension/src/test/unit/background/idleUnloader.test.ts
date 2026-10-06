import { expect, test } from '@jest/globals';
import { shouldUnload } from '../../../background/idleUnloader';

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
