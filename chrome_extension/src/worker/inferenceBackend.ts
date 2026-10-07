/**
 * The onnxruntime-web build the worker runs the model with, chosen once per
 * worker from the user's compute settings: onnxruntime reads its thread count
 * and GPU adapter once, so the service worker starts a new worker to change them.
 *
 *   GPU  the WebGPU build. Its processor fallback is slower than the plain
 *        build's (367 against 320 ms per decoding step), so it is only loaded
 *        when the GPU is wanted and the browser offers one.
 *   CPU  the plain WebAssembly build.
 *
 * When the GPU fails, the same build carries on on the processor for the
 * rest of the worker's life: in a worker where a WebGPU session has failed,
 * new WebGPU sessions fail too, and a failed session must never run again
 * (onnxruntime-web then stops responding).
 */
import type * as Ort from 'onnxruntime-web';
import { findGpu, threadsFor, type ComputeSettings, type GpuLike, type GpuProblem, type Processor, type RunningOn } from '../shared/compute';

export type OrtModule = Pick<typeof Ort, 'InferenceSession' | 'LoraAdapter' | 'Tensor' | 'env'>;

export type OrtBuild = 'webgpu' | 'wasm';

/** What the worker's surroundings provide; injectable for tests. */
export interface BackendHost {
  readonly gpu: GpuLike | undefined;
  readonly hardwareConcurrency: number;
  readonly crossOriginIsolated: boolean;
  /** Absolute URL of the folder holding onnxruntime-web's .wasm binaries. */
  readonly wasmBaseUrl: string;
  importOrt(build: OrtBuild): Promise<OrtModule>;
}

export class InferenceBackend {
  private gpuWorked = false;

  private constructor(
    readonly ort: OrtModule,
    private processor: Processor,
    private readonly threads: number,
  ) {}

  static async start(compute: ComputeSettings, host: BackendHost): Promise<InferenceBackend> {
    const threads = threadsFor(compute.cpuUsage, host.hardwareConcurrency, host.crossOriginIsolated);
    const onGpu = compute.processor === 'gpu' && (await findGpu(host.gpu, compute.gpuPower)) !== null;
    const ort = await host.importOrt(onGpu ? 'webgpu' : 'wasm');
    ort.env.wasm.wasmPaths = host.wasmBaseUrl.endsWith('/') ? host.wasmBaseUrl : `${host.wasmBaseUrl}/`;
    ort.env.wasm.numThreads = threads;
    ort.env.wasm.proxy = false; // already off the page's thread: this *is* the worker, and adapters need it off
    if (onGpu) ort.env.webgpu.powerPreference = compute.gpuPower;
    return new InferenceBackend(ort, onGpu ? 'gpu' : 'cpu', threads);
  }

  get runningOn(): RunningOn {
    return { processor: this.processor, threads: this.threads };
  }

  /** Options for the sessions created from now on. */
  sessionOptions(): Ort.InferenceSession.SessionOptions {
    const common = {
      graphOptimizationLevel: 'all',
      // The adapter weights are graph inputs with empty defaults by design; onnxruntime warns about each one.
      logSeverityLevel: 3,
    } as const;
    if (this.processor === 'cpu') return { ...common, executionProviders: ['wasm'] };
    // On the GPU the processor threads mostly wait for it: let them sleep rather than spin (measured: no slower).
    return { ...common, executionProviders: ['webgpu'], extra: { session: { intra_op: { allow_spinning: '0' } } } };
  }

  /** A rewrite finished on the GPU: a later failure is a passing one, not a GPU this model can't use. */
  gpuSucceeded(): void {
    this.gpuWorked = true;
  }

  /**
   * The GPU failed: every session from now on runs on the processor. Returns
   * the problem to remember across restarts when the GPU never produced a
   * rewrite here, and null when it had (a driver reset, the computer waking up).
   */
  gpuFailed(error: unknown): GpuProblem | null {
    this.processor = 'cpu';
    if (this.gpuWorked) return null;
    return { message: error instanceof Error ? error.message : String(error) };
  }
}
