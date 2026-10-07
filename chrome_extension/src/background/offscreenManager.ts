/**
 * Lifecycle of the single offscreen document that hosts the inference worker.
 *
 * `ensure()` is single-flight: two menu clicks in quick succession must not
 * race into two `createDocument` calls (the second would throw). It also
 * survives the service worker restarting while the document lives on: the
 * document is looked up with `runtime.getContexts` rather than remembered.
 * A document being closed is waited for, so work asked for meanwhile goes to
 * the next document instead of to the one going away.
 */
import type { ComputeSettings } from '../shared/compute';
import type { OffscreenCommand } from '../shared/messages';

export const OFFSCREEN_PATH = 'offscreen/offscreen.html';

export interface OffscreenApis {
  readonly offscreen: Pick<typeof chrome.offscreen, 'createDocument' | 'closeDocument'>;
  readonly runtime: Pick<typeof chrome.runtime, 'getContexts' | 'getURL' | 'sendMessage'>;
}

export class OffscreenManager {
  private pending: Promise<void> | null = null;
  private closing: Promise<void> | null = null;

  /** `compute` is read whenever a document is created: its worker runs the model that way for its whole life. */
  constructor(
    private readonly compute: () => Promise<ComputeSettings>,
    private readonly apis: OffscreenApis = { offscreen: chrome.offscreen, runtime: chrome.runtime },
  ) {}

  async exists(): Promise<boolean> {
    const contexts = await this.apis.runtime.getContexts({
      contextTypes: ['OFFSCREEN_DOCUMENT' as chrome.runtime.ContextType],
      documentUrls: [this.apis.runtime.getURL(OFFSCREEN_PATH)],
    });
    return contexts.length > 0;
  }

  ensure(): Promise<void> {
    if (!this.pending) {
      const closing = this.closing;
      this.pending = (async () => {
        await closing?.catch(() => undefined);
        await this.create();
      })().finally(() => {
        this.pending = null;
      });
    }
    return this.pending;
  }

  private async create(): Promise<void> {
    if (await this.exists()) return;
    const compute = await this.compute();
    await this.apis.offscreen.createDocument({
      url: OFFSCREEN_PATH,
      reasons: ['WORKERS' as chrome.offscreen.Reason],
      justification: 'Runs the on-device writing model in a Web Worker, away from the pages you are typing in.',
    });
    // createDocument resolves once the document has loaded, so its listener
    // is registered; the reply proves the round trip actually works.
    try {
      await this.send({ target: 'offscreen', kind: 'engine/start', compute });
    } catch (error) {
      // A document whose worker never started would be reused as it is: close it so the next ensure() starts afresh.
      await this.apis.offscreen.closeDocument().catch(() => undefined);
      throw error;
    }
  }

  /** Close the document (and its worker); a later ensure() opens a new one. */
  close(): Promise<void> {
    if (!this.closing) {
      const pending = this.pending;
      this.closing = (async () => {
        await pending?.catch(() => undefined);
        if (await this.exists()) await this.apis.offscreen.closeDocument();
      })().finally(() => {
        this.closing = null;
      });
    }
    return this.closing;
  }

  async send(command: OffscreenCommand): Promise<void> {
    await this.apis.runtime.sendMessage(command);
  }
}
