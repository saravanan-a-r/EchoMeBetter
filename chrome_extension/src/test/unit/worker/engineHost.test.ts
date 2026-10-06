import { describe, expect, jest, test } from '@jest/globals';
import { parseModelManifest } from '../../../engine/manifest';
import type { RewriteOptions } from '../../../engine/rewriteEngine';
import { EchoError } from '../../../shared/errors';
import type { EngineEvent } from '../../../shared/messages';
import type { StyleId } from '../../../shared/styles';
import { EngineHost, type LoadedEngine } from '../../../worker/engineHost';
import { readJsonFixture } from '../../helpers/fixtures';

const manifest = parseModelManifest(readJsonFixture('tiny-echo/model.json'));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function fakeEngine(rewrite: (style: StyleId, text: string, options?: RewriteOptions) => Promise<string>): LoadedEngine {
  return {
    manifest,
    rewrite: async (style, text, options) => ({ text: await rewrite(style, text, options), inputTokens: 1, outputTokens: 1 }),
    release: async () => undefined,
  };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('EngineHost', () => {
  test('loads the model once for concurrent requests and reports progress to waiting jobs', async () => {
    const events: EngineEvent[] = [];
    const load = deferred<LoadedEngine>();
    let reportProgress!: (fraction: number) => void;
    const loader = jest.fn((onProgress: (fraction: number) => void) => {
      reportProgress = onProgress;
      return load.promise;
    });
    const host = new EngineHost(loader, (event) => events.push(event));

    host.warmUp();
    host.enqueue('job-1', 'concise', 'hello');
    await flush();
    reportProgress(0.5);
    load.resolve(fakeEngine(async (_style, text) => text.toUpperCase()));
    await host.idle();

    expect(loader).toHaveBeenCalledTimes(1);
    expect(events).toContainEqual({ type: 'job-phase', jobId: 'job-1', phase: 'loading-model', progress: 0.5 });
    expect(events).toContainEqual({ type: 'status', status: expect.objectContaining({ state: 'ready' }) });
    expect(events.at(-1)).toEqual({ type: 'job-done', jobId: 'job-1', text: 'HELLO' });
  });

  test('a failed load is reported, fails the job, and can be retried', async () => {
    const events: EngineEvent[] = [];
    const loader = jest
      .fn<(onProgress: (fraction: number) => void) => Promise<LoadedEngine>>()
      .mockRejectedValueOnce(new Error('out of memory'))
      .mockResolvedValueOnce(fakeEngine(async () => 'ok'));
    const host = new EngineHost(loader, (event) => events.push(event));

    host.enqueue('a', 'concise', 'x');
    await host.idle();
    expect(events).toContainEqual({ type: 'status', status: { state: 'error', message: 'out of memory' } });
    expect(events.at(-1)).toMatchObject({ type: 'job-failed', jobId: 'a', error: { code: 'MODEL_LOAD_FAILED' } });

    host.enqueue('b', 'concise', 'x');
    await host.idle();
    expect(events.at(-1)).toEqual({ type: 'job-done', jobId: 'b', text: 'ok' });
  });

  test('jobs run one at a time, in order', async () => {
    const order: string[] = [];
    const host = new EngineHost(
      async () =>
        fakeEngine(async (_style, text) => {
          order.push(`start ${text}`);
          await flush();
          order.push(`end ${text}`);
          return text;
        }),
      () => undefined,
    );
    host.enqueue('1', 'concise', 'a');
    host.enqueue('2', 'concise', 'b');
    await host.idle();
    expect(order).toEqual(['start a', 'end a', 'start b', 'end b']);
  });

  test('a job cancelled while queued never runs', async () => {
    const events: EngineEvent[] = [];
    const rewrite = jest.fn(async () => 'done');
    const host = new EngineHost(async () => fakeEngine(rewrite), (event) => events.push(event));
    host.enqueue('1', 'concise', 'a');
    host.enqueue('2', 'concise', 'b');
    host.cancel('2');
    await host.idle();
    expect(rewrite).toHaveBeenCalledTimes(1);
    expect(events.at(-1)).toMatchObject({ type: 'job-failed', jobId: '2', error: { code: 'CANCELLED' } });
  });

  test('a running job receives the abort signal', async () => {
    const events: EngineEvent[] = [];
    const host = new EngineHost(
      async () =>
        fakeEngine(
          (_style, _text, options) =>
            new Promise((_resolve, reject) => options?.signal?.addEventListener('abort', () => reject(new EchoError('CANCELLED')))),
        ),
      (event) => events.push(event),
    );
    host.enqueue('1', 'concise', 'a');
    await flush();
    await flush();
    host.cancel('1');
    await host.idle();
    expect(events.at(-1)).toMatchObject({ type: 'job-failed', jobId: '1', error: { code: 'CANCELLED' } });
  });
});

describe('EngineHost without a downloaded model', () => {
  test('the job fails as "not downloaded" and the engine reads as unloaded, not broken', async () => {
    const events: EngineEvent[] = [];
    const host = new EngineHost(
      async () => {
        throw new EchoError('MODEL_NOT_DOWNLOADED');
      },
      (event) => events.push(event),
    );
    host.enqueue('j', 'grammar', 'x');
    await host.idle();
    expect(events).toContainEqual({ type: 'status', status: { state: 'unloaded' } });
    expect(events.some((event) => event.type === 'status' && event.status.state === 'error')).toBe(false);
    expect(events.at(-1)).toMatchObject({ type: 'job-failed', jobId: 'j', error: { code: 'MODEL_NOT_DOWNLOADED' } });
  });
});

describe('EngineHost.unload', () => {
  test('lets the running job finish, then frees the model and reports it unloaded', async () => {
    const events: EngineEvent[] = [];
    const finish = deferred<string>();
    const release = jest.fn(async () => undefined);
    const host = new EngineHost(async () => ({ ...fakeEngine(() => finish.promise), release }), (event) => events.push(event));

    host.enqueue('j', 'concise', 'x');
    await flush();
    const unloading = host.unload();
    await flush();
    expect(release).not.toHaveBeenCalled();
    finish.resolve('done');
    await unloading;

    expect(release).toHaveBeenCalledTimes(1);
    const jobDone = events.findIndex((event) => event.type === 'job-done');
    const unloaded = events.findIndex((event) => event.type === 'status' && event.status.state === 'unloaded');
    expect(jobDone).toBeGreaterThan(-1);
    expect(unloaded).toBeGreaterThan(jobDone);
  });

  test('the next job loads the model again', async () => {
    const loader = jest.fn(async () => fakeEngine(async () => 'ok'));
    const host = new EngineHost(loader, () => undefined);
    host.warmUp();
    await host.unload();
    host.enqueue('j', 'concise', 'x');
    await host.idle();
    expect(loader).toHaveBeenCalledTimes(2);
  });

  test('with nothing loaded it is a no-op', async () => {
    const events: EngineEvent[] = [];
    await new EngineHost(async () => fakeEngine(async () => 'ok'), (event) => events.push(event)).unload();
    expect(events).toEqual([]);
  });
});
