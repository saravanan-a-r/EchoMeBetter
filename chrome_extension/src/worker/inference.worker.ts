/**
 * Dedicated Web Worker that owns onnxruntime-web and the model.
 *
 * Spawned by the offscreen document, so inference never runs on any page's
 * thread, nor on the offscreen document's own. It also owns the model's
 * files: downloading, reading and removing them needs the synchronous file
 * handles only a dedicated worker has. Messages in are WorkerRequest;
 * messages out are EngineEvent.
 */
import * as ort from 'onnxruntime-web/wasm';
import type { EngineEvent, WorkerConfig, WorkerRequest } from '../shared/messages';
import { EngineHost } from './engineHost';
import { ModelLibrary } from './modelLibrary';
import { loadEngine } from './modelLoader';
import { sha256Hex } from './modelDownloader';
import { ModelStore, type DirectoryHandleLike } from './modelStore';

interface WorkerScope {
  postMessage(message: EngineEvent): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void;
}

const scope = globalThis as unknown as WorkerScope;
const emit = (event: EngineEvent) => scope.postMessage(event);
let config: WorkerConfig | null = null;
let library: ModelLibrary | null = null;

const host = new EngineHost(async (onProgress) => {
  if (!config || !library) throw new Error('inference worker used before it was configured');
  const { model, store } = await library.require();
  return loadEngine(ort, config, model, store, onProgress);
}, emit);

function configure(next: WorkerConfig): void {
  config = next;
  library = new ModelLibrary(next.modelSourceUrl, {
    openStore: async () => ModelStore.open((await navigator.storage.getDirectory()) as unknown as DirectoryHandleLike),
    transport: {
      fetch: (url, init) => fetch(url, init),
      estimate: () => navigator.storage.estimate(),
      digest: sha256Hex,
      now: () => performance.now(),
    },
    engine: host,
    emit,
  });
  library.inspect().catch((error: unknown) => {
    console.error('EchoMeBetter: could not read the model storage', error);
    emit({ type: 'installed', installed: null });
  });
}

scope.addEventListener('message', ({ data }) => {
  switch (data.type) {
    case 'configure':
      configure(data.config);
      break;
    case 'warm-up':
      host.warmUp();
      break;
    case 'rewrite':
      host.enqueue(data.jobId, data.style, data.text);
      break;
    case 'cancel':
      host.cancel(data.jobId);
      break;
    case 'download':
      void library?.startDownload();
      break;
    case 'cancel-download':
      library?.cancelDownload();
      break;
    case 'remove-model':
      library?.remove().catch((error: unknown) => {
        console.error('EchoMeBetter: could not remove the model', error);
        void library?.inspect();
      });
      break;
  }
});
