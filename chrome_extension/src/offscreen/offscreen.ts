import { MODEL_SOURCE_URL } from '../shared/modelSource';
import { chooseThreadCount, startBridge, type RuntimeLike, type WorkerLike } from './bridge';

const worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), { type: 'module' });

startBridge(
  worker as unknown as WorkerLike,
  chrome.runtime as unknown as RuntimeLike,
  {
    modelSourceUrl: MODEL_SOURCE_URL,
    wasmBaseUrl: chrome.runtime.getURL('ort/'),
    threads: chooseThreadCount(navigator.hardwareConcurrency, globalThis.crossOriginIsolated),
  },
  navigator.storage,
);
