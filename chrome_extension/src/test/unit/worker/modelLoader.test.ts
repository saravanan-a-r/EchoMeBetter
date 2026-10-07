import { describe, expect, jest, test } from '@jest/globals';
import * as ort from 'onnxruntime-web';
import { DEFAULT_COMPUTE, type ComputeSettings, type GpuAdapterLike, type GpuProblem } from '../../../shared/compute';
import type { StyleId } from '../../../shared/styles';
import { EngineBrokenError } from '../../../worker/engineHost';
import { InferenceBackend, type BackendHost, type OrtBuild, type OrtModule } from '../../../worker/inferenceBackend';
import { ModelLibrary } from '../../../worker/modelLibrary';
import { sha256Hex } from '../../../worker/modelDownloader';
import { engineLoader, loadEngine } from '../../../worker/modelLoader';
import { ModelStore } from '../../../worker/modelStore';
import { readJsonFixture, type Parity } from '../../helpers/fixtures';
import { MemoryDirectory } from '../../helpers/memoryFileSystem';
import { FIXTURE_URL, modelServer, tinyModelFiles } from '../../helpers/modelServer';

const parity = readJsonFixture<Parity>('tiny-echo/parity.json');
/** A professional case that ends in EOS, so its rewrite is returned rather than refused as too long. */
const finished = parity.generation.find((c) => c.style === 'professional' && c.outputIds.at(-1) === 1)!;

const CPU: ComputeSettings = { ...DEFAULT_COMPUTE, processor: 'cpu' };

interface GpuBehaviour {
  /** What `navigator.gpu.requestAdapter()` gives; undefined means the browser has no WebGPU at all. */
  adapter?: GpuAdapterLike | null;
  sessionsFail?: boolean;
  /** Runs fail from this run on (counted over every WebGPU session run). */
  runsFailFrom?: number;
}

/**
 * Real onnxruntime sessions, tensors and adapters under Node, behind a fake
 * worker: the environment the backend configures is recorded rather than
 * applied, and "WebGPU" sessions are WebAssembly ones that fail on cue.
 */
function workerHost(gpu: GpuBehaviour = {}) {
  const env = { wasm: {} as Record<string, unknown>, webgpu: {} as Record<string, unknown> };
  const created: ort.LoraAdapter[] = [];
  const sessionOptions: ort.InferenceSession.SessionOptions[] = [];
  const LoraAdapter = {
    create: jest.fn(async (bytes: Uint8Array) => {
      const adapter = await ort.LoraAdapter.create(bytes);
      jest.spyOn(adapter, 'release');
      created.push(adapter);
      return adapter;
    }),
  };
  let gpuRuns = 0;
  const InferenceSession = {
    create: async (bytes: Uint8Array, options: ort.InferenceSession.SessionOptions) => {
      sessionOptions.push(options);
      if (options.executionProviders?.[0] !== 'webgpu') return ort.InferenceSession.create(bytes, options);
      if (gpu.sessionsFail) throw new Error('WebGPU: buffer size exceeds the limit');
      const session = await ort.InferenceSession.create(bytes, { ...options, executionProviders: ['wasm'], extra: undefined });
      const run = session.run.bind(session);
      session.run = ((...args: Parameters<typeof run>) => {
        gpuRuns += 1;
        if (gpu.runsFailFrom !== undefined && gpuRuns >= gpu.runsFailFrom) return Promise.reject(new Error('GPU device was lost'));
        return run(...args);
      }) as typeof session.run;
      return session;
    },
  };
  const module = { InferenceSession, Tensor: ort.Tensor, LoraAdapter, env } as unknown as OrtModule;
  const importOrt = jest.fn(async (_build: OrtBuild) => module);
  const host: BackendHost = {
    gpu: gpu.adapter === undefined ? undefined : { requestAdapter: async () => gpu.adapter ?? null },
    hardwareConcurrency: 8,
    crossOriginIsolated: true,
    wasmBaseUrl: 'chrome-extension://id/ort',
    importOrt,
  };
  return { host, env, LoraAdapter, created, sessionOptions, importOrt };
}

const APPLE_GPU: GpuAdapterLike = { info: { vendor: 'apple' } };

async function installed(adapters: StyleId[]) {
  const library = new ModelLibrary(FIXTURE_URL, {
    openStore: (() => {
      const root = new MemoryDirectory();
      return () => ModelStore.open(root);
    })(),
    transport: { fetch: modelServer(tinyModelFiles('tiny-echo')).fetch, estimate: async () => ({}), digest: sha256Hex, now: () => 0 },
    engine: { unload: async () => undefined },
    emit: () => undefined,
  });
  await library.startDownload(adapters);
  return library;
}

const noProblems = (problem: GpuProblem) => {
  throw new Error(`unexpected GPU problem: ${problem.message}`);
};

describe('loadEngine', () => {
  test('the base loads from storage with progress; each rewrite loads its adapter and releases it after', async () => {
    const library = await installed(['professional']);
    const { host, env, LoraAdapter, created, importOrt } = workerHost();
    const fractions: number[] = [];
    const backend = await InferenceBackend.start(CPU, host);
    const engine = await loadEngine(backend, library, (fraction) => fractions.push(fraction), noProblems);

    expect(importOrt.mock.calls).toEqual([['wasm']]);
    expect(env.wasm).toEqual({ wasmPaths: 'chrome-extension://id/ort/', numThreads: 4, proxy: false });
    expect(engine.runningOn).toEqual({ processor: 'cpu', threads: 4 });
    expect(fractions.at(-1)).toBe(1);
    expect(fractions).toEqual([...fractions].sort((a, b) => a - b));
    expect(LoraAdapter.create).not.toHaveBeenCalled();

    expect((await engine.rewrite('professional', finished.input)).text).toBe(finished.outputText.trim());
    // A style still in training rewrites with the fallback adapter.
    expect((await engine.rewrite('concise', finished.input)).text).toBe(finished.outputText.trim());
    // An encoder and a decoder half per rewrite, each released once the rewrite is done.
    expect(created).toHaveLength(4);
    for (const adapter of created) expect(adapter.release).toHaveBeenCalledTimes(1);
    await engine.release();
  });

  test('a style whose adapter is not downloaded is refused; one added while the base is loaded is used at once', async () => {
    const library = await installed(['professional']);
    const { host, LoraAdapter } = workerHost();
    const engine = await loadEngine(await InferenceBackend.start(CPU, host), library, () => undefined, noProblems);
    await expect(engine.rewrite('grammar', finished.input)).rejects.toMatchObject({
      code: 'STYLE_NOT_DOWNLOADED',
      details: { style: 'Grammar', adapter: 'Grammar' },
    });

    await library.startDownload(['grammar']);
    // The fixture's grammar rewrite of this text runs past maxNewTokens, so it is refused as
    // too long -- after running with the grammar adapter.
    await expect(engine.rewrite('grammar', finished.input)).rejects.toMatchObject({ code: 'OUTPUT_TOO_LONG' });
    const grammarFiles = tinyModelFiles('tiny-echo');
    expect(LoraAdapter.create.mock.calls.map(([bytes]) => bytes)).toEqual([
      grammarFiles.get('adapters/grammar/encoder.onnx_adapter'),
      grammarFiles.get('adapters/grammar/decoder.onnx_adapter'),
    ]);
    await engine.release();
  });

  test('a base file that went missing after the download fails the load as "not downloaded"', async () => {
    const library = await installed([]);
    const { store, model } = await library.require();
    await store.remove(model.manifest.files.encoder);
    const { host } = workerHost();
    await expect(loadEngine(await InferenceBackend.start(CPU, host), library, () => undefined, noProblems)).rejects.toMatchObject({ code: 'MODEL_NOT_DOWNLOADED' });
  });
});

describe('on the GPU', () => {
  test('with a GPU on offer, the WebGPU build runs the model there, with the chosen power preference', async () => {
    const library = await installed(['professional']);
    const { host, env, importOrt, sessionOptions } = workerHost({ adapter: APPLE_GPU });
    const engine = await loadEngine(await InferenceBackend.start({ ...DEFAULT_COMPUTE, gpuPower: 'low-power' }, host), library, () => undefined, noProblems);

    expect(importOrt.mock.calls).toEqual([['webgpu']]);
    expect(env.webgpu).toEqual({ powerPreference: 'low-power' });
    // Its processor threads are still set: they run what the GPU hands back, and the processor fallback.
    expect(env.wasm).toMatchObject({ numThreads: 4 });
    expect(sessionOptions.map((options) => options.executionProviders)).toEqual([['webgpu'], ['webgpu']]);
    expect(sessionOptions[0]).toMatchObject({ extra: { session: { intra_op: { allow_spinning: '0' } } } });
    expect(engine.runningOn.processor).toBe('gpu');
    expect((await engine.rewrite('professional', finished.input)).text).toBe(finished.outputText.trim());
    await engine.release();
  });

  test.each([
    ['the browser has no WebGPU', undefined],
    ['it offers no adapter', null],
    ['it offers only a software adapter', { info: { vendor: 'google', isFallbackAdapter: true } }],
    ['it offers only a software adapter (older Chrome)', { isFallbackAdapter: true }],
  ])('when %s, the processor build is used', async (_label, adapter) => {
    const { host, importOrt } = workerHost({ adapter });
    const backend = await InferenceBackend.start(DEFAULT_COMPUTE, host);
    expect(importOrt.mock.calls).toEqual([['wasm']]);
    expect(backend.runningOn.processor).toBe('cpu');
  });

  test('chosen off by the user, the GPU is not even looked for', async () => {
    const requestAdapter = jest.fn(async () => APPLE_GPU);
    const { host, importOrt } = workerHost();
    await InferenceBackend.start(CPU, { ...host, gpu: { requestAdapter } });
    expect(requestAdapter).not.toHaveBeenCalled();
    expect(importOrt.mock.calls).toEqual([['wasm']]);
  });

  test('sessions the GPU cannot hold are created on the processor instead, and the GPU is reported as not working', async () => {
    const library = await installed(['professional']);
    const { host, sessionOptions } = workerHost({ adapter: APPLE_GPU, sessionsFail: true });
    const problems: GpuProblem[] = [];
    const fractions: number[] = [];
    const engine = await loadEngine(await InferenceBackend.start(DEFAULT_COMPUTE, host), library, (fraction) => fractions.push(fraction), (problem) => problems.push(problem));

    expect(problems).toEqual([{ message: 'WebGPU: buffer size exceeds the limit' }]);
    expect(sessionOptions.map((options) => options.executionProviders)).toEqual([['webgpu'], ['wasm'], ['wasm']]);
    expect(engine.runningOn.processor).toBe('cpu');
    expect(fractions.at(-1)).toBe(1);
    expect((await engine.rewrite('professional', finished.input)).text).toBe(finished.outputText.trim());
    await engine.release();
  });

  test('a GPU that fails its first rewrite breaks the engine and is reported; the next load runs on the processor', async () => {
    const library = await installed(['professional']);
    const { host, created, importOrt } = workerHost({ adapter: APPLE_GPU, runsFailFrom: 1 });
    const problems: GpuProblem[] = [];
    const load = engineLoader(() => InferenceBackend.start(DEFAULT_COMPUTE, host), library, (problem) => problems.push(problem));

    const gpuEngine = await load(() => undefined);
    const failure = await gpuEngine.rewrite('professional', finished.input).catch((error: unknown) => error);
    expect(failure).toBeInstanceOf(EngineBrokenError);
    expect(problems).toEqual([{ message: 'GPU device was lost' }]);
    // The adapter is released even though the rewrite failed.
    for (const adapter of created) expect(adapter.release).toHaveBeenCalledTimes(1);

    const cpuEngine = await load(() => undefined);
    expect(cpuEngine.runningOn.processor).toBe('cpu');
    expect((await cpuEngine.rewrite('professional', finished.input)).text).toBe(finished.outputText.trim());
    expect(importOrt).toHaveBeenCalledTimes(1);
    await Promise.all([gpuEngine.release(), cpuEngine.release()]);
  });

  test('a GPU that fails after it has worked breaks the engine, but is not reported: the next start tries it again', async () => {
    const library = await installed(['professional']);
    // The first rewrite takes one encoder run and a decoder run per token; fail well after it.
    const { host } = workerHost({ adapter: APPLE_GPU, runsFailFrom: finished.outputIds.length + 2 });
    const problems: GpuProblem[] = [];
    const engine = await loadEngine(await InferenceBackend.start(DEFAULT_COMPUTE, host), library, () => undefined, (problem) => problems.push(problem));
    expect((await engine.rewrite('professional', finished.input)).text).toBe(finished.outputText.trim());
    await expect(engine.rewrite('professional', finished.input)).rejects.toBeInstanceOf(EngineBrokenError);
    expect(problems).toEqual([]);
    await engine.release();
  });

  test('errors of the rewrite itself are not taken for the GPU failing', async () => {
    const library = await installed(['professional']);
    const { host } = workerHost({ adapter: APPLE_GPU });
    const engine = await loadEngine(await InferenceBackend.start(DEFAULT_COMPUTE, host), library, () => undefined, noProblems);
    await expect(engine.rewrite('grammar', finished.input)).rejects.toMatchObject({ code: 'STYLE_NOT_DOWNLOADED' });
    expect(engine.runningOn.processor).toBe('gpu');
    await engine.release();
  });
});

test('a backend that failed to start (its onnxruntime build did not load) is started again by the next load', async () => {
  const library = await installed(['professional']);
  const { host } = workerHost();
  const start = jest
    .fn<() => Promise<InferenceBackend>>()
    .mockRejectedValueOnce(new Error('Failed to fetch dynamically imported module'))
    .mockImplementation(() => InferenceBackend.start(CPU, host));
  const load = engineLoader(start, library, noProblems);
  await expect(load(() => undefined)).rejects.toThrow('Failed to fetch dynamically imported module');
  const engine = await load(() => undefined);
  await load(() => undefined).then((again) => again.release());
  expect(start).toHaveBeenCalledTimes(2);
  await engine.release();
});
