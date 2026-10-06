import { describe, expect, test } from '@jest/globals';
import { parseModelManifest, totalModelBytes } from '../../../engine/manifest';
import { readJsonFixture } from '../../helpers/fixtures';

const valid = readJsonFixture<Record<string, unknown>>('tiny-t5/model.json');
const without = (key: string) => Object.fromEntries(Object.entries(valid).filter(([k]) => k !== key));

describe('parseModelManifest', () => {
  test('accepts a manifest written by the exporter', () => {
    const manifest = parseModelManifest(valid);
    expect(manifest.ioContract).toBe('t5-cross-kv/1');
    expect(totalModelBytes(manifest)).toBeGreaterThan(0);
  });

  test.each([
    ['an unknown schema version', { ...valid, schemaVersion: 2 }, /schemaVersion 2 is not supported/],
    ['a different I/O contract', { ...valid, ioContract: 'other/1' }, /ioContract other\/1 is not supported/],
    ['a style missing from the menu catalogue', { ...valid, styles: { professional: (valid.styles as Record<string, unknown>).professional } }, /styles\.grammar is missing/],
    ['a prompt with no input segment', { ...valid, styles: { ...(valid.styles as object), concise: { prompt: [{ text: 'hi' }] } } }, /exactly one \{"input": true\}/],
    ['a file path escaping the model folder', { ...valid, files: { ...(valid.files as object), encoder: { path: '../x.onnx', bytes: 1, sha256: 'a' } } }, /plain file name/],
    ['an unknown input transform', { ...valid, inputTransforms: ['rot13'] }, /inputTransforms/],
    ['missing limits', without('limits'), /limits must be an object/],
  ])('rejects %s', (_label, manifest, message) => {
    expect(() => parseModelManifest(manifest)).toThrow(message);
  });
});
