/**
 * Dedicated Web Worker that owns onnxruntime-web and the model.
 *
 * Spawned by the offscreen document, so inference never runs on any page's
 * thread, nor on the offscreen document's own. It also owns the model's
 * files (the base and the style adapters): downloading, reading and removing
 * them needs the synchronous file handles only a dedicated worker has.
 * onnxruntime-web is imported on the first model load, in the build the
 * compute settings call for (see inferenceBackend.ts).
 * Messages in are WorkerRequest; messages out are EngineEvent.
 */
import type { GpuLike } from '../shared/compute';
import type { EngineEvent, WorkerConfig, WorkerRequest } from '../shared/messages';
import { EngineHost, type EngineLoader } from './engineHost';
import { InferenceBackend, type OrtBuild, type OrtModule } from './inferenceBackend';
import { ModelLibrary } from './modelLibrary';
import { engineLoader } from './modelLoader';
import { sha256Hex } from './modelDownloader';
import { ModelStore, type DirectoryHandleLike } from './modelStore';

interface WorkerScope {
  postMessage(message: EngineEvent): void;
  addEventListener(type: 'message', listener: (event: MessageEvent<WorkerRequest>) => void): void;
}

const scope = globalThis as unknown as WorkerScope;
const emit = (event: EngineEvent) => scope.postMessage(event);
let library: ModelLibrary | null = null;
let load: EngineLoader | null = null;

const importOrt = (build: OrtBuild): Promise<OrtModule> => (build === 'webgpu' ? import('onnxruntime-web/webgpu') : import('onnxruntime-web/wasm'));

const host = new EngineHost((onProgress) => {
  if (!load) throw new Error('inference worker used before it was configured');
  return load(onProgress);
}, emit);

function configure(config: WorkerConfig): void {
  library = new ModelLibrary(config.modelSourceUrl, {
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
  const backendHost = {
    gpu: (navigator as { gpu?: GpuLike }).gpu,
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated: globalThis.crossOriginIsolated,
    wasmBaseUrl: config.wasmBaseUrl,
    importOrt,
  };
  load = engineLoader(() => InferenceBackend.start(config.compute, backendHost), library, (problem) => emit({ type: 'gpu-problem', problem }));
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
      void library?.startDownload(data.adapters);
      break;
    case 'cancel-download':
      library?.cancelDownload();
      break;
    case 'remove-adapter':
      library?.removeAdapter(data.adapter).catch((error: unknown) => {
        console.error('EchoMeBetter: could not remove the style', error);
        void library?.inspect();
      });
      break;
    case 'remove-model':
      library?.remove().catch((error: unknown) => {
        console.error('EchoMeBetter: could not remove the model', error);
        void library?.inspect();
      });
      break;
  }
});
