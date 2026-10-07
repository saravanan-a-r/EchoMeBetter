/**
 * Build the engine from the downloaded model.
 *
 * Runs inside the inference worker. The base model is read from the
 * ModelStore (the extension's private file system) with byte-level progress
 * and turned into sessions one file at a time, so only one file's bytes are
 * held alongside a session. It then stays in memory until the idle unloader
 * frees it.
 *
 * A style's adapter is loaded only for the rewrite that needs it and released
 * as soon as that rewrite ends: adapters are small and quick to load, and
 * this keeps memory at the base model's size between rewrites. Which adapter
 * a style runs with is looked up from the installed model at each rewrite,
 * so adapters added or removed while the base is loaded take effect at once.
 *
 * On the GPU, a failure inside onnxruntime is taken as the GPU's. Sessions
 * that can't be created there are created on the processor instead. A
 * rewrite that fails leaves sessions that must not run again: the engine
 * throws EngineBrokenError, and the host loads a new one (on the processor)
 * and runs the job again.
 */
import type * as Ort from 'onnxruntime-web';
import { baseFiles, bytesOf, catalogOf, type AdapterManifest, type ModelManifest } from '../engine/manifest';
import type { AdapterPair, TensorFactory } from '../engine/onnx/t5Runner';
import { RewriteEngine } from '../engine/rewriteEngine';
import type { GpuProblem } from '../shared/compute';
import { EchoError } from '../shared/errors';
import { adapterForStyle } from '../shared/modelInstall';
import { styleLabel } from '../shared/styles';
import { EngineBrokenError, type EngineLoader, type LoadedEngine } from './engineHost';
import type { InferenceBackend, OrtModule } from './inferenceBackend';
import type { ModelStore, StoredModel } from './modelStore';

/** Where the engine finds the installed model, as it is at the moment of asking. */
export interface InstalledModelSource {
  require(): Promise<{ model: StoredModel; store: ModelStore }>;
}

/** Told when the GPU can't run the model at all, so later starts go straight to the processor. */
export type GpuProblemReporter = (problem: GpuProblem) => void;

export interface LoadedAdapter extends AdapterPair {
  release(): Promise<void>;
}

export async function loadAdapter(ort: Pick<OrtModule, 'LoraAdapter'>, store: ModelStore, adapter: AdapterManifest): Promise<LoadedAdapter> {
  const encoder = await ort.LoraAdapter.create(await store.read(adapter.files.encoder));
  try {
    const decoder = await ort.LoraAdapter.create(await store.read(adapter.files.decoder));
    return { encoder, decoder, release: async () => void (await Promise.all([encoder.release(), decoder.release()])) };
  } catch (error) {
    await encoder.release();
    throw error;
  }
}

/** Marks an error raised by onnxruntime while it ran on the GPU. */
class GpuFailure extends Error {
  constructor(readonly reason: unknown) {
    super(reason instanceof Error ? reason.message : String(reason));
    this.name = 'GpuFailure';
  }
}

function leaveGpu(backend: InferenceBackend, error: unknown, report: GpuProblemReporter): void {
  const problem = backend.gpuFailed(error);
  if (problem) report(problem);
}

/** The encoder and decoder sessions, each file read just before its session is made. */
async function createSessions(backend: InferenceBackend, store: ModelStore, manifest: ModelManifest, tick: (count: number) => void) {
  const onGpu = backend.runningOn.processor === 'gpu';
  const options = backend.sessionOptions();
  const sessions: Ort.InferenceSession[] = [];
  try {
    for (const file of [manifest.files.encoder, manifest.files.decoder]) {
      const bytes = await store.read(file, tick);
      try {
        sessions.push(await backend.ort.InferenceSession.create(bytes, options));
      } catch (error) {
        throw onGpu ? new GpuFailure(error) : error;
      }
    }
  } catch (error) {
    await Promise.all(sessions.map((session) => session.release()));
    throw error;
  }
  return { encoder: sessions[0]!, decoder: sessions[1]! };
}

export async function loadEngine(
  backend: InferenceBackend,
  source: InstalledModelSource,
  onProgress: (fraction: number) => void,
  reportGpuProblem: GpuProblemReporter,
): Promise<LoadedEngine> {
  const { ort } = backend;
  const { model, store } = await source.require();
  const { manifest } = model;
  const total = bytesOf(baseFiles(manifest));
  let loaded = 0;
  const tick = (count: number) => {
    loaded += count;
    onProgress(Math.min(1, loaded / total));
  };

  const tokenizerJson: unknown = JSON.parse(new TextDecoder().decode(await store.read(manifest.files.tokenizer, tick)));
  let sessions: Awaited<ReturnType<typeof createSessions>>;
  try {
    sessions = await createSessions(backend, store, manifest, tick);
  } catch (error) {
    if (!(error instanceof GpuFailure)) throw error;
    leaveGpu(backend, error.reason, reportGpuProblem);
    loaded = manifest.files.tokenizer.bytes; // read again for the processor; progress is never reported backwards
    sessions = await createSessions(backend, store, manifest, tick);
  }
  const runningOn = backend.runningOn;
  const onGpu = runningOn.processor === 'gpu';
  const engine = new RewriteEngine({
    geometry: manifest.architecture,
    tokenizerJson,
    ...sessions,
    TensorCtor: ort.Tensor as unknown as TensorFactory,
  });

  return {
    manifest,
    runningOn,
    async rewrite(style, text, rewriteOptions) {
      const current = await source.require();
      const adapterId = adapterForStyle(catalogOf(current.model.manifest), style);
      if (!adapterId) throw new EchoError('STYLE_UNAVAILABLE', { style: styleLabel(style) });
      if (!current.model.adapters.includes(adapterId)) {
        throw new EchoError('STYLE_NOT_DOWNLOADED', { style: styleLabel(style), adapter: styleLabel(adapterId) });
      }
      const adapter = await loadAdapter(ort, current.store, current.model.manifest.adapters[adapterId]!);
      try {
        const result = await engine.rewrite({ manifest: current.model.manifest, adapterId, adapter, text }, rewriteOptions);
        if (onGpu) backend.gpuSucceeded();
        return result;
      } catch (error) {
        if (!onGpu || error instanceof EchoError) throw error;
        leaveGpu(backend, error, reportGpuProblem);
        throw new EngineBrokenError(error);
      } finally {
        await adapter.release();
      }
    },
    release: () => engine.release(),
  };
}

/**
 * The host's loader: the backend is started on the first load (so a worker
 * that only downloads never loads onnxruntime), and kept for later ones.
 */
export function engineLoader(startBackend: () => Promise<InferenceBackend>, source: InstalledModelSource, reportGpuProblem: GpuProblemReporter): EngineLoader {
  let backend: Promise<InferenceBackend> | null = null;
  return async (onProgress) => {
    const starting = (backend ??= startBackend());
    let started: InferenceBackend;
    try {
      started = await starting;
    } catch (error) {
      if (backend === starting) backend = null; // e.g. the onnxruntime chunk failed to load: the next load tries again
      throw error;
    }
    return loadEngine(started, source, onProgress, reportGpuProblem);
  };
}
