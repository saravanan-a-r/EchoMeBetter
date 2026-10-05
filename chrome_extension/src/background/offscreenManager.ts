/**
 * Lifecycle of the single offscreen document that hosts the inference worker.
 *
 * `ensure()` is single-flight: two menu clicks in quick succession must not
 * race into two `createDocument` calls (the second would throw). It also
 * survives the service worker restarting while the document lives on: the
 * document is looked up with `runtime.getContexts` rather than remembered.
 */
import type { OffscreenCommand } from '../shared/messages';

export const OFFSCREEN_PATH = 'offscreen/offscreen.html';

export interface OffscreenApis {
  readonly offscreen: Pick<typeof chrome.offscreen, 'createDocument' | 'closeDocument'>;
  readonly runtime: Pick<typeof chrome.runtime, 'getContexts' | 'getURL' | 'sendMessage'>;
}

export class OffscreenManager {
  private pending: Promise<void> | null = null;

  constructor(private readonly apis: OffscreenApis = { offscreen: chrome.offscreen, runtime: chrome.runtime }) {}

  async exists(): Promise<boolean> {
    const contexts = await this.apis.runtime.getContexts({
      contextTypes: ['OFFSCREEN_DOCUMENT' as chrome.runtime.ContextType],
      documentUrls: [this.apis.runtime.getURL(OFFSCREEN_PATH)],
    });
    return contexts.length > 0;
  }

  ensure(): Promise<void> {
    if (!this.pending) {
      this.pending = this.create().finally(() => {
        this.pending = null;
      });
    }
    return this.pending;
  }

  private async create(): Promise<void> {
    if (await this.exists()) return;
    await this.apis.offscreen.createDocument({
      url: OFFSCREEN_PATH,
      reasons: ['WORKERS' as chrome.offscreen.Reason],
      justification: 'Runs the on-device writing model in a Web Worker, away from the pages you are typing in.',
    });
    // createDocument resolves once the document has loaded, so its listener
    // is registered; the ping proves the round trip actually works.
    await this.send({ target: 'offscreen', kind: 'engine/ping' });
  }

  async close(): Promise<void> {
    if (this.pending) await this.pending.catch(() => undefined);
    if (await this.exists()) await this.apis.offscreen.closeDocument();
  }

  async send(command: OffscreenCommand): Promise<void> {
    await this.apis.runtime.sendMessage(command);
  }
}
