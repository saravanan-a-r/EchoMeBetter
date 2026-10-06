import { describe, expect, test } from '@jest/globals';
import { describeError, EchoError, toErrorPayload, type ErrorCode } from '../../../shared/errors';
import { isForegroundMessage, isJobEvent, isJobRequest, isShortcutRequest, isUiRequest } from '../../../shared/messages';
import { isDownloadState, isInstalledModelRecord, modelFrom } from '../../../shared/modelInstall';
import { MODEL_SOURCE_URL, parseModelSourceUrl } from '../../../shared/modelSource';
import { DEFAULT_SETTINGS, parseSettings } from '../../../shared/settings';
import { isEngineStatus } from '../../../shared/status';

const ALL_CODES: ErrorCode[] = [
  'NO_SELECTION', 'UNSUPPORTED_FIELD', 'PASSWORD_FIELD', 'INPUT_TOO_LONG', 'RESERVED_MARKUP', 'TEXT_CHANGED', 'FRAME_INACCESSIBLE',
  'BUSY', 'MODEL_NOT_DOWNLOADED', 'MODEL_LOAD_FAILED', 'INFERENCE_FAILED', 'EMPTY_RESULT', 'OUTPUT_TOO_LONG', 'CANCELLED',
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
    expect(isUiRequest({ kind: 'ui/download-model' })).toBe(true);
    expect(isUiRequest({ kind: 'ui/format-disk' })).toBe(false);
  });
});

describe('settings', () => {
  test('only known values are accepted', () => {
    expect(parseSettings({ keepModelLoadedMinutes: 60, shortcutsEnabled: false })).toEqual({ keepModelLoadedMinutes: 60, shortcutsEnabled: false });
    expect(parseSettings({ keepModelLoadedMinutes: 7, shortcutsEnabled: 'yes' })).toEqual(DEFAULT_SETTINGS);
    expect(parseSettings(undefined)).toEqual(DEFAULT_SETTINGS);
  });

  test('settings saved before shortcuts existed keep their choices and get shortcuts on', () => {
    expect(parseSettings({ keepModelLoadedMinutes: 60 })).toEqual({ keepModelLoadedMinutes: 60, shortcutsEnabled: true });
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
    ['http://localhost:47615/flan/', 'http://localhost:47615/flan/'],
    ['http://127.0.0.1:8080/m/', 'http://127.0.0.1:8080/m/'],
  ])('accepts %s', (raw, expected) => {
    expect(parseModelSourceUrl(raw)).toBe(expected);
  });

  test.each([
    ['plain http from another host', 'http://cdn.example.com/m/', /https/],
    ['a file rather than a folder', 'https://cdn.example.com/m/model.json', /end with "\/"/],
    ['a query string', 'https://cdn.example.com/m/?token=1', /query/],
    ['not a URL', 'models/flan/', /not a valid URL/],
    ['not a string', 42, /non-empty string/],
  ])('rejects %s', (_label, raw, message) => {
    expect(() => parseModelSourceUrl(raw)).toThrow(message);
  });
});

describe('model install state', () => {
  const record = { sourceUrl: 'https://a.example/m/', model: { id: 'x', displayName: 'X', placeholder: false, precision: 'int8', sizeBytes: 1 } };

  test('guards accept what the worker reports and reject anything else', () => {
    expect(isInstalledModelRecord(record)).toBe(true);
    expect(isInstalledModelRecord({ model: {} })).toBe(false);
    expect(isDownloadState({ state: 'downloading', phase: 'fetching', receivedBytes: 1, totalBytes: 2, bytesPerSecond: 0 })).toBe(true);
    expect(isDownloadState({ state: 'failed', error: { code: 'DOWNLOAD_FAILED' } })).toBe(true);
    expect(isDownloadState({ state: 'failed' })).toBe(false);
    expect(isDownloadState({ state: 'paused' })).toBe(false);
  });

  test('a model downloaded from another URL is not the current model', () => {
    expect(modelFrom(record, 'https://a.example/m/')).toBe(record);
    expect(modelFrom(record, 'https://b.example/m/')).toBeNull();
    expect(modelFrom(null, 'https://a.example/m/')).toBeNull();
  });
});
