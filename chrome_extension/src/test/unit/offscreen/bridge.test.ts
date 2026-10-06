import { describe, expect, jest, test } from '@jest/globals';
import { chooseThreadCount, startBridge, type RuntimeLike, type WorkerLike } from '../../../offscreen/bridge';
import type { EngineEvent, OffscreenEventMessage, WorkerRequest } from '../../../shared/messages';

function harness() {
  const toWorker: WorkerRequest[] = [];
  const toBackground: OffscreenEventMessage[] = [];
  const workerListeners: Record<string, (event: never) => void> = {};
  let runtimeListener!: Parameters<RuntimeLike['onMessage']['addListener']>[0];
  const worker: WorkerLike = {
    postMessage: (message) => toWorker.push(message),
    addEventListener: (type: string, listener: (event: never) => void) => {
      workerListeners[type] = listener;
    },
  } as WorkerLike;
  const runtime: RuntimeLike = {
    sendMessage: async (message) => {
      toBackground.push(message);
    },
    onMessage: { addListener: (listener) => (runtimeListener = listener) },
  };
  const storage = { persist: jest.fn(async () => true) };
  startBridge(worker, runtime, { modelSourceUrl: 'https://models.example.test/m/', wasmBaseUrl: 'ext://ort/', threads: 2 }, storage);
  const send = (message: unknown) => {
    const replies: unknown[] = [];
    runtimeListener(message, {}, (reply) => replies.push(reply));
    return replies;
  };
  const emit = (event: EngineEvent) => workerListeners.message!({ data: event } as never);
  return { toWorker, toBackground, send, emit, workerListeners, storage };
}

describe('offscreen bridge', () => {
  test('configures the worker first', () => {
    expect(harness().toWorker[0]).toEqual({
      type: 'configure',
      config: { modelSourceUrl: 'https://models.example.test/m/', wasmBaseUrl: 'ext://ort/', threads: 2 },
    });
  });

  test('relays service-worker commands to the worker and answers pings', () => {
    const { send, toWorker } = harness();
    expect(send({ target: 'offscreen', kind: 'engine/ping' })).toEqual([{ ok: true }]);
    send({ target: 'offscreen', kind: 'engine/rewrite', jobId: 'j', style: 'concise', text: 'hi' });
    send({ target: 'offscreen', kind: 'engine/cancel', jobId: 'j' });
    expect(toWorker.slice(1)).toEqual([
      { type: 'rewrite', jobId: 'j', style: 'concise', text: 'hi' },
      { type: 'cancel', jobId: 'j' },
    ]);
  });

  test('relays model commands, asking for persistent storage before a download', async () => {
    const { send, toWorker, storage } = harness();
    let grant!: (granted: boolean) => void;
    storage.persist.mockImplementationOnce(() => new Promise<boolean>((resolve) => (grant = resolve)));
    send({ target: 'offscreen', kind: 'model/download' });
    // Sent while the download still waits on the persistence request: must not overtake it.
    send({ target: 'offscreen', kind: 'model/cancel-download' });
    send({ target: 'offscreen', kind: 'model/remove' });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toWorker).toHaveLength(1);
    grant(true);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(storage.persist).toHaveBeenCalledTimes(1);
    expect(toWorker.slice(1)).toEqual([{ type: 'download' }, { type: 'cancel-download' }, { type: 'remove-model' }]);
  });

  test('a refused persistence request does not stop the download', async () => {
    const { send, toWorker, storage } = harness();
    storage.persist.mockRejectedValueOnce(new Error('not allowed'));
    send({ target: 'offscreen', kind: 'model/download' });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(toWorker.at(-1)).toEqual({ type: 'download' });
  });

  test('ignores messages meant for other contexts', () => {
    const { send, toWorker } = harness();
    expect(send({ kind: 'ui/warm-up' })).toEqual([]);
    expect(toWorker).toHaveLength(1);
  });

  test('forwards engine events, and a crashed worker becomes an error status', () => {
    const { emit, toBackground, workerListeners } = harness();
    emit({ type: 'job-done', jobId: 'j', text: 'out' });
    workerListeners.error!({ message: 'boom' } as never);
    expect(toBackground).toEqual([
      { target: 'background', kind: 'engine/event', event: { type: 'job-done', jobId: 'j', text: 'out' } },
      { target: 'background', kind: 'engine/event', event: { type: 'status', status: { state: 'error', message: 'boom' } } },
    ]);
  });
});

test.each([
  [8, true, 4],
  [4, true, 2],
  [1, true, 1],
  [16, true, 4],
  [8, false, 1],
])('chooseThreadCount(%i cores, isolated=%s) = %i', (cores, isolated, threads) => {
  expect(chooseThreadCount(cores, isolated)).toBe(threads);
});
