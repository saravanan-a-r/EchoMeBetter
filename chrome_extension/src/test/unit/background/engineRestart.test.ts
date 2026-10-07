import { describe, expect, jest, test } from '@jest/globals';
import { EngineRestarter, readEngineCompute, recordEngineCompute, RESTART_PENDING_KEY, type RestartDeps } from '../../../background/engineRestart';
import { DEFAULT_COMPUTE, type ComputeSettings } from '../../../shared/compute';

function memoryStorage() {
  const data = new Map<string, unknown>();
  return {
    data,
    get: async (key: string) => (data.has(key) ? { [key]: data.get(key) } : {}),
    set: async (items: Record<string, unknown>) => void Object.entries(items).forEach(([key, value]) => data.set(key, value)),
    remove: async (key: string) => void data.delete(key),
  } as unknown as RestartDeps['storage'] & { data: Map<string, unknown> };
}

const CPU: ComputeSettings = { ...DEFAULT_COMPUTE, processor: 'cpu' };

async function setup(options: { running?: ComputeSettings | null; wanted?: ComputeSettings; hostRunning?: boolean; busy?: boolean } = {}) {
  const storage = memoryStorage();
  if (options.running !== null) await recordEngineCompute(options.running ?? DEFAULT_COMPUTE, storage);
  const state = { wanted: options.wanted ?? CPU, hostRunning: options.hostRunning ?? true, busy: options.busy ?? false, freeAtRestart: true };
  const restart = jest.fn(async () => state.freeAtRestart);
  const restarter = new EngineRestarter({
    storage,
    wanted: async () => state.wanted,
    hostRunning: async () => state.hostRunning,
    busy: async () => state.busy,
    restart,
  });
  return { storage, state, restart, restarter, pending: () => storage.data.get(RESTART_PENDING_KEY) === true };
}

describe('EngineRestarter', () => {
  test('an idle engine started with other settings is restarted at once', async () => {
    const { restarter, restart, pending } = await setup();
    await restarter.request();
    expect(restart).toHaveBeenCalledTimes(1);
    expect(pending()).toBe(false);
  });

  test('with no engine running there is nothing to restart: the next one starts with the new settings', async () => {
    const { restarter, restart, pending } = await setup({ hostRunning: false });
    await restarter.request();
    expect(restart).not.toHaveBeenCalled();
    expect(pending()).toBe(false);
  });

  test('settings changed and changed back leave the engine alone', async () => {
    const { restarter, restart } = await setup({ running: CPU, wanted: CPU });
    await restarter.request();
    expect(restart).not.toHaveBeenCalled();
  });

  test('the GPU power setting means nothing to an engine on the processor', async () => {
    const { restarter, restart } = await setup({ running: CPU, wanted: { ...CPU, gpuPower: 'low-power' } });
    await restarter.request();
    expect(restart).not.toHaveBeenCalled();
  });

  test('a busy engine is restarted once it is free, and only once', async () => {
    const { restarter, restart, state, pending } = await setup({ busy: true });
    await restarter.request();
    expect(restart).not.toHaveBeenCalled();
    expect(pending()).toBe(true);
    await restarter.settle();
    expect(restart).not.toHaveBeenCalled();

    state.busy = false;
    await Promise.all([restarter.settle(), restarter.settle()]);
    expect(restart).toHaveBeenCalledTimes(1);
    expect(pending()).toBe(false);
    await restarter.settle();
    expect(restart).toHaveBeenCalledTimes(1);
  });

  test('a job that arrives just before the restart postpones it to the next settle', async () => {
    const { restarter, restart, state } = await setup();
    state.freeAtRestart = false;
    await restarter.request();
    state.freeAtRestart = true;
    await restarter.settle();
    expect(restart).toHaveBeenCalledTimes(2);
  });

  test('an engine of unknown settings (started by an older version) is restarted', async () => {
    const { restarter, restart } = await setup({ running: null });
    await restarter.request();
    expect(restart).toHaveBeenCalledTimes(1);
  });

  test('a failing restart is logged and leaves the restart owed', async () => {
    const { restarter, restart, pending } = await setup();
    const log = jest.spyOn(console, 'error').mockImplementation(() => undefined);
    restart.mockRejectedValueOnce(new Error('closeDocument failed'));
    await restarter.request();
    expect(log).toHaveBeenCalled();
    expect(pending()).toBe(true);
    await restarter.settle();
    expect(restart).toHaveBeenCalledTimes(2);
    expect(pending()).toBe(false);
  });
});

test('the recorded engine settings are read back, and anything malformed reads as unknown', async () => {
  const storage = memoryStorage();
  expect(await readEngineCompute(storage)).toBeNull();
  await recordEngineCompute(CPU, storage);
  expect(await readEngineCompute(storage)).toEqual(CPU);
  await storage.set({ engineCompute: { processor: 'tpu', cpuUsage: 'balanced', gpuPower: 'low-power' } });
  expect(await readEngineCompute(storage)).toBeNull();
});
