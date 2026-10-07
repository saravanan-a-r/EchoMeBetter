import { describe, expect, test } from '@jest/globals';
import { describeError, EchoError, toErrorPayload, type ErrorCode } from '../../../shared/errors';
import { isForegroundMessage, isJobEvent, isJobRequest, isShortcutRequest, isUiRequest } from '../../../shared/messages';
import { canRewrite, isDownloadState, isInstalledModelRecord, modelFrom, offeredCatalog, styleProblem, styleReadiness } from '../../../shared/modelInstall';
import { installedRecord, TINY_CATALOG } from '../../helpers/fixtures';
import { MODEL_SOURCE_URL, parseModelSourceUrl } from '../../../shared/modelSource';
import { computeToRun, DEFAULT_COMPUTE, findGpu, parseGpuProblem, sameCompute, threadsFor } from '../../../shared/compute';
import { DEFAULT_SETTINGS, parseSettings } from '../../../shared/settings';
import { isEngineStatus } from '../../../shared/status';

const ALL_CODES: ErrorCode[] = [
  'NO_SELECTION', 'UNSUPPORTED_FIELD', 'PASSWORD_FIELD', 'INPUT_TOO_LONG', 'RESERVED_MARKUP', 'TEXT_CHANGED', 'FRAME_INACCESSIBLE',
  'BUSY', 'MODEL_NOT_DOWNLOADED', 'STYLE_NOT_DOWNLOADED', 'STYLE_UNAVAILABLE', 'MODEL_OUTDATED', 'MODEL_LOAD_FAILED', 'INFERENCE_FAILED', 'EMPTY_RESULT', 'OUTPUT_TOO_LONG', 'CANCELLED',
  'DOWNLOAD_FAILED', 'STORAGE_FULL', 'DISCONNECTED', 'INTERNAL',
];

describe('errors', () => {
  test('every code has user-facing copy', () => {
    for (const code of ALL_CODES) expect(describeError({ code }).length).toBeGreaterThan(10);
  });

  test('details are woven into the message', () => {
    expect(describeError({ code: 'INPUT_TOO_LONG', details: { actual: 900, limit: 512 } })).toContain('900 of 512');
  });

  test('anything thrown becomes a payload that can cross a message boundary', () => {
    expect(toErrorPayload(new EchoError('BUSY'))).toEqual({ code: 'BUSY', details: {} });
    expect(toErrorPayload(new Error('x'), 'INFERENCE_FAILED')).toEqual({ code: 'INFERENCE_FAILED', details: { message: 'x' } });
    expect(JSON.parse(JSON.stringify(toErrorPayload('weird')))).toEqual({ code: 'INTERNAL', details: { message: 'weird' } });
  });
});

describe('message guards', () => {
  test('accept well-formed messages and reject everything else', () => {
    expect(isForegroundMessage({ kind: 'echo/start-job', jobId: 'j', style: 'concise' })).toBe(true);
    expect(isForegroundMessage({ kind: 'echo/start-job', jobId: 'j', style: 'loud' })).toBe(false);
    expect(isJobRequest({ kind: 'job/request', jobId: 'j', style: 'grammar', text: 'x' })).toBe(true);
    expect(isJobRequest({ kind: 'job/request', jobId: 'j', style: 'grammar' })).toBe(false);
    expect(isJobEvent({ kind: 'job/done', jobId: 'j', text: 'x' })).toBe(true);
    expect(isJobEvent(null)).toBe(false);
    expect(isUiRequest({ kind: 'ui/download-model', adapters: ['professional'] })).toBe(true);
    expect(isUiRequest({ kind: 'ui/download-model', adapters: ['casual'] })).toBe(false);
    expect(isUiRequest({ kind: 'ui/download-model' })).toBe(false);
    expect(isUiRequest({ kind: 'ui/remove-adapter', adapter: 'grammar' })).toBe(true);
    expect(isUiRequest({ kind: 'ui/remove-adapter' })).toBe(false);
    expect(isUiRequest({ kind: 'ui/format-disk' })).toBe(false);
  });
});

describe('settings', () => {
  test('only known values are accepted', () => {
    const chosen = { keepModelLoadedMinutes: 60, shortcutsEnabled: false, processor: 'cpu', cpuUsage: 'maximum', gpuPower: 'low-power' };
    expect(parseSettings(chosen)).toEqual(chosen);
    expect(parseSettings({ keepModelLoadedMinutes: 7, shortcutsEnabled: 'yes', processor: 'tpu', cpuUsage: 'all', gpuPower: 'turbo' })).toEqual(DEFAULT_SETTINGS);
    expect(parseSettings(undefined)).toEqual(DEFAULT_SETTINGS);
  });

  test('the GPU is preferred, with balanced processor use and full GPU power', () => {
    expect(DEFAULT_SETTINGS).toMatchObject({ processor: 'gpu', cpuUsage: 'balanced', gpuPower: 'high-performance' });
  });

  test('settings saved by an older version keep their choices and get the defaults for what is new', () => {
    expect(parseSettings({ keepModelLoadedMinutes: 60 })).toEqual({ ...DEFAULT_SETTINGS, keepModelLoadedMinutes: 60 });
    expect(parseSettings({ keepModelLoadedMinutes: 5, shortcutsEnabled: false })).toEqual({ ...DEFAULT_SETTINGS, keepModelLoadedMinutes: 5, shortcutsEnabled: false });
  });
});

describe('compute', () => {
  test.each([
    // cores, light, balanced, maximum
    [1, 1, 1, 1],
    [2, 1, 1, 1],
    [4, 1, 2, 3],
    [8, 2, 4, 5],
    [10, 2, 4, 6],
    [16, 2, 4, 8],
    [64, 2, 4, 8],
  ])('%i cores: light %i, balanced %i, maximum %i threads', (cores, light, balanced, maximum) => {
    expect([threadsFor('light', cores, true), threadsFor('balanced', cores, true), threadsFor('maximum', cores, true)]).toEqual([light, balanced, maximum]);
  });

  test('without cross-origin isolation there is one thread, and an unknown core count counts as two', () => {
    expect(threadsFor('maximum', 16, false)).toBe(1);
    expect(threadsFor('balanced', 0, true)).toBe(1);
  });

  test('a GPU is one the browser offers that is not rendered in software', async () => {
    const offering = (adapter: unknown) => ({ requestAdapter: async () => adapter as never });
    expect(await findGpu(undefined)).toBeNull();
    expect(await findGpu(offering(null))).toBeNull();
    expect(await findGpu(offering({ info: { vendor: 'google', isFallbackAdapter: true } }))).toBeNull();
    expect(await findGpu(offering({ isFallbackAdapter: true, info: { vendor: '' } }))).toBeNull();
    expect(await findGpu({ requestAdapter: async () => Promise.reject(new Error('blocked')) })).toBeNull();
    expect(await findGpu(offering({ info: { vendor: 'nvidia' } }))).toEqual({ vendor: 'nvidia' });
    expect(await findGpu(offering({}))).toEqual({ vendor: '' });
  });

  test('a recorded GPU problem keeps the engine on the processor, and only engine-relevant changes count', () => {
    expect(computeToRun(DEFAULT_COMPUTE, null)).toEqual(DEFAULT_COMPUTE);
    expect(computeToRun({ ...DEFAULT_SETTINGS }, { message: 'x' })).toEqual({ ...DEFAULT_COMPUTE, processor: 'cpu' });
    expect(sameCompute(DEFAULT_COMPUTE, { ...DEFAULT_COMPUTE, gpuPower: 'low-power' })).toBe(false);
    expect(sameCompute({ ...DEFAULT_COMPUTE, processor: 'cpu' }, { ...DEFAULT_COMPUTE, processor: 'cpu', gpuPower: 'low-power' })).toBe(true);
    expect(sameCompute(DEFAULT_COMPUTE, { ...DEFAULT_COMPUTE, cpuUsage: 'light' })).toBe(false);
    expect(parseGpuProblem({ message: 'm' })).toEqual({ message: 'm' });
    expect(parseGpuProblem({ reason: 'm' })).toBeNull();
    expect(parseGpuProblem(undefined)).toBeNull();
  });
});

test('shortcut request guard', () => {
  expect(isShortcutRequest({ kind: 'shortcut/rewrite', style: 'grammar' })).toBe(true);
  expect(isShortcutRequest({ kind: 'shortcut/rewrite', style: 'shouting' })).toBe(false);
  expect(isShortcutRequest({ kind: 'echo/start-job', style: 'grammar' })).toBe(false);
});

test('engine status guard', () => {
  expect(isEngineStatus({ state: 'loading', progress: 0.3 })).toBe(true);
  expect(isEngineStatus({ state: 'sleeping' })).toBe(false);
});

describe('model source', () => {
  test('the configured URL is valid', () => {
    expect(parseModelSourceUrl(MODEL_SOURCE_URL)).toBe(MODEL_SOURCE_URL);
  });

  test.each([
    ['https://cdn.example.com/models/echomebetter-v1/', 'https://cdn.example.com/models/echomebetter-v1/'],
    ['http://localhost:47615/echomebetter/', 'http://localhost:47615/echomebetter/'],
    ['http://127.0.0.1:8080/m/', 'http://127.0.0.1:8080/m/'],
  ])('accepts %s', (raw, expected) => {
    expect(parseModelSourceUrl(raw)).toBe(expected);
  });

  test.each([
    ['plain http from another host', 'http://cdn.example.com/m/', /https/],
    ['a file rather than a folder', 'https://cdn.example.com/m/model.json', /end with "\/"/],
    ['a query string', 'https://cdn.example.com/m/?token=1', /query/],
    ['not a URL', 'models/echomebetter/', /not a valid URL/],
    ['not a string', 42, /non-empty string/],
  ])('rejects %s', (_label, raw, message) => {
    expect(() => parseModelSourceUrl(raw)).toThrow(message);
  });
});

describe('model install state', () => {
  const record = installedRecord('https://a.example/m/');
  const target = { base: true, adapters: ['professional'] };

  test('guards accept what the worker reports and reject anything else', () => {
    expect(isInstalledModelRecord(record)).toBe(true);
    expect(isInstalledModelRecord({ sourceUrl: 'x', model: {} })).toBe(false);
    expect(isInstalledModelRecord({ ...record, adapters: ['casual'] })).toBe(false);
    expect(isDownloadState({ state: 'downloading', target, phase: 'fetching', receivedBytes: 1, totalBytes: 2, bytesPerSecond: 0 })).toBe(true);
    expect(isDownloadState({ state: 'downloading', phase: 'fetching', receivedBytes: 1, totalBytes: 2, bytesPerSecond: 0 })).toBe(false);
    expect(isDownloadState({ state: 'failed', target, error: { code: 'DOWNLOAD_FAILED' } })).toBe(true);
    expect(isDownloadState({ state: 'failed', target })).toBe(false);
    expect(isDownloadState({ state: 'paused' })).toBe(false);
  });

  test('a model downloaded from another URL is not the current model', () => {
    expect(modelFrom(record, 'https://a.example/m/')).toBe(record);
    expect(modelFrom(record, 'https://b.example/m/')).toBeNull();
    expect(modelFrom(null, 'https://a.example/m/')).toBeNull();
  });

  test('a style runs with its own adapter, or the fallback until it has one', () => {
    expect(styleProblem('grammar', installedRecord('x', ['grammar']))).toBeNull();
    expect(styleProblem('concise', installedRecord('x', ['professional']))).toBeNull();
    expect(styleProblem('grammar', installedRecord('x', ['professional']))).toEqual({
      code: 'STYLE_NOT_DOWNLOADED',
      details: { style: 'Grammar', adapter: 'Grammar' },
    });
    expect(styleProblem('concise', installedRecord('x', ['grammar']))).toEqual({
      code: 'STYLE_NOT_DOWNLOADED',
      details: { style: 'Concise', adapter: 'Professional' },
    });
    expect(styleProblem('concise', null)).toEqual({ code: 'MODEL_NOT_DOWNLOADED' });
    const noFallback = { ...record, catalog: { ...TINY_CATALOG, fallbackAdapter: null } };
    expect(styleProblem('concise', noFallback)).toEqual({ code: 'STYLE_UNAVAILABLE', details: { style: 'Concise' } });
  });

  test('something can be rewritten once the base and one adapter are installed', () => {
    expect(canRewrite(null)).toBe(false);
    expect(canRewrite(installedRecord('x', []))).toBe(false);
    expect(canRewrite(installedRecord('x', ['grammar']))).toBe(true);
  });

  test('readiness says what each style still needs', () => {
    const professional = TINY_CATALOG.adapters.find((adapter) => adapter.style === 'professional')!;
    const grammar = TINY_CATALOG.adapters.find((adapter) => adapter.style === 'grammar')!;
    expect(styleReadiness('concise', TINY_CATALOG, null)).toEqual({ state: 'needs-model', adapter: professional });
    expect(styleReadiness('grammar', TINY_CATALOG, installedRecord('x', ['professional']))).toEqual({ state: 'needs-download', adapter: grammar });
    expect(styleReadiness('elaborate', TINY_CATALOG, installedRecord('x', ['professional']))).toEqual({ state: 'ready', adapter: 'professional' });
  });

  test('downloads are offered from the server only while it serves the installed base', () => {
    const newer = { ...TINY_CATALOG, base: 'another' };
    expect(offeredCatalog(null, newer)).toBe(newer);
    expect(offeredCatalog(record, newer)).toBe(record.catalog);
    expect(offeredCatalog(record, TINY_CATALOG)).toBe(TINY_CATALOG);
    expect(offeredCatalog(record, null)).toBe(record.catalog);
  });
});
