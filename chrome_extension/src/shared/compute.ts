/**
 * Where the model runs and how much of the computer it may use.
 *
 * The graphics chip (WebGPU) is the default wherever the browser offers a
 * real one: on an Apple M4 a decoding step takes about 215 ms there against
 * about 320 ms on four processor threads, and encoding a long prompt is
 * three times faster. A GPU sits idle on most desktops, so it gets the
 * high-performance adapter unless the user asks to save power.
 *
 * The processor is shared with everything else the user is doing, so its use
 * is a choice of three levels. Measured on the same machine (10 cores): one
 * thread 690 ms per step, two 460 ms, four 320 ms, six 310 ms, all ten 385 ms.
 * Decoding stops gaining at about four threads and using every core is
 * slower than using some, so even "maximum" leaves cores free.
 */
export const PROCESSORS = ['gpu', 'cpu'] as const;
export type Processor = (typeof PROCESSORS)[number];

export const CPU_USAGES = ['light', 'balanced', 'maximum'] as const;
export type CpuUsage = (typeof CPU_USAGES)[number];

/** Passed to `navigator.gpu.requestAdapter()`; it picks between two graphics chips where a computer has both. */
export const GPU_POWERS = ['high-performance', 'low-power'] as const;
export type GpuPower = (typeof GPU_POWERS)[number];

export interface ComputeSettings {
  /** The preferred processor. The GPU is used only where the browser offers one that works. */
  readonly processor: Processor;
  readonly cpuUsage: CpuUsage;
  readonly gpuPower: GpuPower;
}

export const DEFAULT_COMPUTE: ComputeSettings = { processor: 'gpu', cpuUsage: 'balanced', gpuPower: 'high-performance' };

/** Whether an engine started with `a` runs as one started with `b` would. The GPU's power setting means nothing on the processor. */
export function sameCompute(a: ComputeSettings, b: ComputeSettings): boolean {
  return a.processor === b.processor && a.cpuUsage === b.cpuUsage && (a.processor === 'cpu' || a.gpuPower === b.gpuPower);
}

export function isComputeSettings(value: unknown): value is ComputeSettings {
  if (typeof value !== 'object' || value === null) return false;
  const compute = value as Partial<Record<keyof ComputeSettings, unknown>>;
  return (
    (PROCESSORS as readonly unknown[]).includes(compute.processor) &&
    (CPU_USAGES as readonly unknown[]).includes(compute.cpuUsage) &&
    (GPU_POWERS as readonly unknown[]).includes(compute.gpuPower)
  );
}

/**
 * Inference threads for a usage level. Threads need SharedArrayBuffer, i.e. a
 * cross-origin isolated context; without it there is exactly one.
 */
export function threadsFor(usage: CpuUsage, hardwareConcurrency: number, crossOriginIsolated: boolean): number {
  if (!crossOriginIsolated) return 1;
  const cores = Math.max(1, Math.floor(hardwareConcurrency) || 2);
  switch (usage) {
    case 'light':
      return Math.max(1, Math.min(2, Math.floor(cores / 4)));
    case 'balanced':
      return Math.max(1, Math.min(4, Math.floor(cores / 2)));
    case 'maximum':
      return Math.max(1, Math.min(8, cores - 1, Math.ceil(cores * 0.6)));
  }
}

/** The model as it runs right now: reported once it is loaded. */
export interface RunningOn {
  readonly processor: Processor;
  /** Processor threads; on the GPU they only serve the few steps it hands back. */
  readonly threads: number;
}

export function isRunningOn(value: unknown): value is RunningOn {
  if (typeof value !== 'object' || value === null) return false;
  const running = value as { processor?: unknown; threads?: unknown };
  return (PROCESSORS as readonly unknown[]).includes(running.processor) && typeof running.threads === 'number';
}

// ---------------------------------------------------------------------------
// Finding a usable GPU
// ---------------------------------------------------------------------------

/** The slice of WebGPU this needs; `navigator.gpu` in pages and workers. */
export interface GpuLike {
  requestAdapter(options?: { powerPreference?: GpuPower }): Promise<GpuAdapterLike | null>;
}

export interface GpuAdapterLike {
  readonly info?: { readonly vendor?: string; readonly isFallbackAdapter?: boolean };
  /** Older Chrome versions report this on the adapter rather than in `info`. */
  readonly isFallbackAdapter?: boolean;
}

export interface GpuInfo {
  /** Lower-case vendor such as "apple", "intel" or "nvidia"; empty when the browser keeps it to itself. */
  readonly vendor: string;
}

/**
 * The GPU the browser would hand onnxruntime, or null when there is none or
 * only a software one (which renders on the processor and is slower than
 * running the model there directly).
 */
export async function findGpu(gpu: GpuLike | undefined, powerPreference: GpuPower = 'high-performance'): Promise<GpuInfo | null> {
  if (!gpu) return null;
  try {
    const adapter = await gpu.requestAdapter({ powerPreference });
    if (!adapter || adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true) return null;
    return { vendor: adapter.info?.vendor ?? '' };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// A GPU that could not run the model
// ---------------------------------------------------------------------------

/**
 * Recorded when the GPU fails the model before it ever produced a rewrite
 * (the browser offers a GPU this model does not run on). Until it is cleared
 * the model loads straight onto the processor, instead of failing on the
 * GPU first every time. A GPU that fails after it had worked (the driver
 * restarted, the computer woke up) is not recorded: the next load tries it again.
 */
export interface GpuProblem {
  readonly message: string;
}

export const GPU_PROBLEM_STORAGE_KEY = 'gpuProblem';

export function parseGpuProblem(value: unknown): GpuProblem | null {
  if (typeof value !== 'object' || value === null) return null;
  const { message } = value as { message?: unknown };
  return typeof message === 'string' ? { message } : null;
}

export async function readGpuProblem(storage: chrome.storage.StorageArea = chrome.storage.local): Promise<GpuProblem | null> {
  return parseGpuProblem((await storage.get(GPU_PROBLEM_STORAGE_KEY))[GPU_PROBLEM_STORAGE_KEY]);
}

export async function writeGpuProblem(problem: GpuProblem | null, storage: chrome.storage.StorageArea = chrome.storage.local): Promise<void> {
  if (problem) await storage.set({ [GPU_PROBLEM_STORAGE_KEY]: problem });
  else await storage.remove(GPU_PROBLEM_STORAGE_KEY);
}

/** What the engine is started with: the user's choice, on the processor while the GPU has a recorded problem. */
export function computeToRun(settings: ComputeSettings, problem: GpuProblem | null): ComputeSettings {
  return { processor: problem ? 'cpu' : settings.processor, cpuUsage: settings.cpuUsage, gpuPower: settings.gpuPower };
}
