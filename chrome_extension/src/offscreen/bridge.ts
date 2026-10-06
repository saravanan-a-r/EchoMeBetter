/**
 * The offscreen document's only job: host the inference worker and relay
 * messages between it and the service worker.
 *
 * Service workers cannot start dedicated workers, and onnxruntime-web needs
 * one (plus cross-origin isolation for its threads), so this hidden
 * extension page exists purely to give the worker a home that outlives any
 * single tab.
 *
 * Before a download it also asks for persistent storage (a window-only API),
 * so the browser does not evict the downloaded model when disk space runs low.
 */
import { isOffscreenCommand, type EngineEvent, type OffscreenEventMessage, type WorkerConfig, type WorkerRequest } from '../shared/messages';

export interface WorkerLike {
  postMessage(message: WorkerRequest): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<EngineEvent>) => void): void;
  addEventListener(type: 'error', listener: (event: ErrorEvent) => void): void;
}

export interface RuntimeLike {
  sendMessage(message: OffscreenEventMessage): Promise<unknown>;
  onMessage: {
    addListener(
      listener: (message: unknown, sender: unknown, sendResponse: (response?: unknown) => void) => boolean | undefined,
    ): void;
  };
}

/**
 * Inference threads: half the cores, at most four, at least one -- the rest
 * stay free for the page the user is typing into. Threads need
 * SharedArrayBuffer, i.e. a cross-origin isolated document.
 */
export function chooseThreadCount(hardwareConcurrency: number, crossOriginIsolated: boolean): number {
  if (!crossOriginIsolated) return 1;
  return Math.max(1, Math.min(4, Math.floor((hardwareConcurrency || 2) / 2)));
}

export interface StorageLike {
  persist(): Promise<boolean>;
}

export function startBridge(worker: WorkerLike, runtime: RuntimeLike, config: WorkerConfig, storage: StorageLike): void {
  const forward = (event: EngineEvent) => {
    runtime.sendMessage({ target: 'background', kind: 'engine/event', event }).catch(() => {
      // The service worker may be between lifetimes; it re-reads state on wake.
    });
  };

  worker.addEventListener('message', ({ data }) => forward(data));
  worker.addEventListener('error', (event) => {
    forward({ type: 'status', status: { state: 'error', message: event.message || 'inference worker crashed' } });
  });
  worker.postMessage({ type: 'configure', config });

  // Model commands reach the worker in the order they were sent, even while
  // a download is still waiting on the persistence request.
  let modelCommands: Promise<unknown> = Promise.resolve();
  const inOrder = (step: () => Promise<unknown> | void) => {
    modelCommands = modelCommands.then(step);
  };

  runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!isOffscreenCommand(message)) return undefined;
    switch (message.kind) {
      case 'engine/ping':
        sendResponse({ ok: true });
        return undefined;
      case 'engine/warm-up':
        worker.postMessage({ type: 'warm-up' });
        break;
      case 'engine/rewrite':
        worker.postMessage({ type: 'rewrite', jobId: message.jobId, style: message.style, text: message.text });
        break;
      case 'engine/cancel':
        worker.postMessage({ type: 'cancel', jobId: message.jobId });
        break;
      case 'model/download':
        // Not granted is fine: the model then lives in best-effort storage, and is downloaded again if evicted.
        inOrder(() =>
          storage
            .persist()
            .catch(() => false)
            .then(() => worker.postMessage({ type: 'download', adapters: message.adapters })),
        );
        break;
      case 'model/cancel-download':
        inOrder(() => worker.postMessage({ type: 'cancel-download' }));
        break;
      case 'model/remove-adapter':
        inOrder(() => worker.postMessage({ type: 'remove-adapter', adapter: message.adapter }));
        break;
      case 'model/remove':
        inOrder(() => worker.postMessage({ type: 'remove-model' }));
        break;
    }
    sendResponse({ ok: true });
    return undefined;
  });
}
