/**
 * Applies new compute settings to a running engine.
 *
 * onnxruntime fixes its thread count and GPU when it starts, so the engine
 * host (the offscreen document and its worker) is closed and opened again.
 * Never while a rewrite or a download needs it: the restart then waits for
 * them, and is retried each time one of them ends. A model that was loaded
 * is loaded again straight away, so the next rewrite does not wait for it.
 *
 * What the running engine was started with, and whether a restart is still
 * owed, live in `chrome.storage.session`: the service worker may be suspended
 * between the change and the moment the engine is free.
 */
import { isComputeSettings, sameCompute, type ComputeSettings } from '../shared/compute';

export const ENGINE_COMPUTE_KEY = 'engineCompute';
export const RESTART_PENDING_KEY = 'engineRestartPending';

export interface RestartDeps {
  readonly storage: Pick<chrome.storage.StorageArea, 'get' | 'set' | 'remove'>;
  /** The settings an engine started now would get. */
  wanted(): Promise<ComputeSettings>;
  hostRunning(): Promise<boolean>;
  /** A rewrite or a download is using the engine. */
  busy(): Promise<boolean>;
  /** Close the engine host and load the model again if it was loaded; false when it turned out to be busy after all. */
  restart(): Promise<boolean>;
}

/** Remember what a newly started engine runs with. */
export async function recordEngineCompute(compute: ComputeSettings, storage: RestartDeps['storage'] = chrome.storage.session): Promise<void> {
  await storage.set({ [ENGINE_COMPUTE_KEY]: compute });
}

export async function readEngineCompute(storage: RestartDeps['storage'] = chrome.storage.session): Promise<ComputeSettings | null> {
  const value = (await storage.get(ENGINE_COMPUTE_KEY))[ENGINE_COMPUTE_KEY];
  return isComputeSettings(value) ? value : null;
}

export class EngineRestarter {
  /** One check at a time, so two events never restart the engine twice. */
  private queue: Promise<void> = Promise.resolve();

  constructor(private readonly deps: RestartDeps) {}

  /** The settings changed: restart now if the engine is free, else as soon as it is. */
  request(): Promise<void> {
    return this.enqueue(async () => {
      await this.deps.storage.set({ [RESTART_PENDING_KEY]: true });
      await this.attempt();
    });
  }

  /** Something that may have held a restart back has finished. */
  settle(): Promise<void> {
    return this.enqueue(() => this.attempt());
  }

  private enqueue(step: () => Promise<void>): Promise<void> {
    this.queue = this.queue.then(step).catch((error: unknown) => console.error('EchoMeBetter: could not apply the new settings', error));
    return this.queue;
  }

  private async attempt(): Promise<void> {
    const { storage } = this.deps;
    if ((await storage.get(RESTART_PENDING_KEY))[RESTART_PENDING_KEY] !== true) return;
    const running = await readEngineCompute(storage);
    // No engine: the next one starts with the new settings. Changed and changed back: nothing to do.
    if (!(await this.deps.hostRunning()) || (running && sameCompute(running, await this.deps.wanted()))) {
      await storage.remove(RESTART_PENDING_KEY);
      return;
    }
    if (await this.deps.busy()) return;
    if (await this.deps.restart()) await storage.remove(RESTART_PENDING_KEY);
  }
}
