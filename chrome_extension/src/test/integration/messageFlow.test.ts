/**
 * The production message path with only Chrome's transport faked:
 *
 *   page port ⇄ JobRouter (service worker) ⇄ runtime messages ⇄ offscreen bridge ⇄ worker messages ⇄ EngineHost
 *
 * The model is a stand-in, but every message shape, every hop and the
 * cancel path are the real ones -- this catches a protocol change made on
 * one side and not the other.
 */
import { describe, expect, test } from '@jest/globals';
import { JobRouter, type PortLike } from '../../background/jobRouter';
import { startBridge, type RuntimeLike, type WorkerLike } from '../../offscreen/bridge';
import { parseModelManifest } from '../../engine/manifest';
import { EchoError } from '../../shared/errors';
import { isOffscreenEventMessage, type EngineEvent, type JobEvent, type OffscreenCommand, type WorkerRequest } from '../../shared/messages';
import { EngineHost, type LoadedEngine } from '../../worker/engineHost';
import { readJsonFixture } from '../helpers/fixtures';

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function wire(engine: LoadedEngine) {
  // worker <-> offscreen (postMessage is asynchronous in the browser)
  const workerToBridge: ((event: { data: EngineEvent }) => void)[] = [];
  const host = new EngineHost(async () => engine, (event) => setTimeout(() => workerToBridge.forEach((listener) => listener({ data: event }))));
  const worker: WorkerLike = {
    postMessage: (request: WorkerRequest) =>
      setTimeout(() => {
        if (request.type === 'rewrite') host.enqueue(request.jobId, request.style, request.text);
        if (request.type === 'cancel') host.cancel(request.jobId);
      }),
    addEventListener: (type: string, listener: (event: never) => void) => {
      if (type === 'message') workerToBridge.push(listener as never);
    },
  } as WorkerLike;

  // offscreen <-> service worker (chrome.runtime.sendMessage)
  let offscreenListener!: Parameters<RuntimeLike['onMessage']['addListener']>[0];
  let router!: JobRouter;
  const runtime: RuntimeLike = {
    sendMessage: async (message) => {
      if (isOffscreenEventMessage(message)) router.handleEngineEvent(message.event);
    },
    onMessage: { addListener: (listener) => (offscreenListener = listener) },
  };
  startBridge(worker, runtime, { modelSourceUrl: 'https://models.example.test/m/', wasmBaseUrl: 'o/', threads: 1 }, { persist: async () => true });
  router = new JobRouter({
    ensureEngineHost: async () => undefined,
    sendToEngine: async (command: OffscreenCommand) => void offscreenListener(command, {}, () => undefined),
    onActivity: () => undefined,
  });

  // page <-> service worker (a Port)
  const received: JobEvent[] = [];
  let toRouter!: (message: unknown) => void;
  let disconnect!: () => void;
  const port: PortLike = {
    name: 'echo/job',
    postMessage: (message) => received.push(message),
    onMessage: { addListener: (listener) => (toRouter = listener) },
    onDisconnect: { addListener: (listener) => (disconnect = listener) },
  };
  router.handlePort(port);
  return { received, send: (message: unknown) => toRouter(message), disconnect: () => disconnect(), host };
}

const manifest = parseModelManifest(readJsonFixture('tiny-t5/model.json'));

describe('page → service worker → offscreen → worker and back', () => {
  test('a job comes back as phases then the rewritten text', async () => {
    const { received, send, host } = wire({
      manifest,
      rewrite: async (style, text) => ({ text: `[${style}] ${text}`, inputTokens: 1, outputTokens: 1 }),
      release: async () => undefined,
    });
    send({ kind: 'job/request', jobId: 'J', style: 'friendly', text: 'hello' });
    for (let i = 0; i < 10; i++) await tick();
    await host.idle();
    for (let i = 0; i < 5; i++) await tick();
    expect(received.map((event) => event.kind)).toEqual(['job/phase', 'job/phase', 'job/done']);
    expect(received.at(-1)).toEqual({ kind: 'job/done', jobId: 'J', text: '[friendly] hello' });
  });

  test('closing the tab mid-rewrite aborts the generation in the worker', async () => {
    let aborted = false;
    const { send, disconnect, host } = wire({
      manifest,
      rewrite: (_style, _text, options) =>
        new Promise((_resolve, reject) =>
          options?.signal?.addEventListener('abort', () => {
            aborted = true;
            reject(new EchoError('CANCELLED'));
          }),
        ),
      release: async () => undefined,
    });
    send({ kind: 'job/request', jobId: 'J', style: 'concise', text: 'long text' });
    for (let i = 0; i < 10; i++) await tick();
    disconnect();
    for (let i = 0; i < 10; i++) await tick();
    await host.idle();
    expect(aborted).toBe(true);
  });
});
