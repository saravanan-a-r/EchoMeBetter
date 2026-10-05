/**
 * What happens when the user picks a style from the right-click menu.
 *
 * 1. Inject the foreground script into the exact frame that was clicked --
 *    only now, on demand. Nothing of EchoMeBetter runs on a page the user
 *    never asks it to touch.
 * 2. Start the model loading in parallel with the page's own setup, so the
 *    first rewrite does not pay for both one after the other.
 * 3. Tell the frame to start the job; from there the page owns the UI and
 *    talks to the service worker over its job port.
 *
 * Before any of that costs anything, a model that has not been downloaded
 * yet is reported in the page instead (pointing at the toolbar popup).
 *
 * The context-menu click grants `activeTab`, which is what allows the
 * injection without any host permissions. A cross-origin frame is outside
 * that grant; then the top frame shows why nothing happened.
 */
import type { ErrorPayload } from '../shared/errors';
import type { ForegroundMessage } from '../shared/messages';
import { parseStyleMenuId } from './contextMenus';

export const FOREGROUND_SCRIPT = 'foreground.js';

export interface LauncherDeps {
  readonly scripting: Pick<typeof chrome.scripting, 'executeScript'>;
  readonly tabs: { sendMessage(tabId: number, message: ForegroundMessage, options: { frameId: number }): Promise<unknown> };
  readonly action: Pick<typeof chrome.action, 'setBadgeText' | 'setBadgeBackgroundColor' | 'setTitle'>;
  /** URL prefix of this extension's own pages, which load the foreground themselves. */
  readonly extensionOrigin: string;
  readonly prepareEngine: () => void;
  /** Whether the model is downloaded, as far as the service worker knows. */
  readonly modelReady: () => Promise<boolean>;
  readonly newJobId: () => string;
}

export interface MenuClick {
  readonly menuItemId: string | number;
  readonly frameId?: number;
}

async function inject(deps: LauncherDeps, tabId: number, frameId: number): Promise<void> {
  await deps.scripting.executeScript({ target: { tabId, frameIds: [frameId] }, files: [FOREGROUND_SCRIPT] });
}

async function flagUnreachable(deps: LauncherDeps, tabId: number): Promise<void> {
  // Last resort when not even the top frame can be scripted (e.g. the Web
  // Store or a PDF viewer): say so on the toolbar icon.
  await deps.action.setBadgeBackgroundColor({ tabId, color: '#E5484D' });
  await deps.action.setBadgeText({ tabId, text: '!' });
  await deps.action.setTitle({ tabId, title: "EchoMeBetter can't edit text on this page." });
}

async function notice(deps: LauncherDeps, tabId: number, error: ErrorPayload): Promise<void> {
  try {
    await inject(deps, tabId, 0);
    await deps.tabs.sendMessage(tabId, { kind: 'echo/notice', error }, { frameId: 0 });
  } catch {
    await flagUnreachable(deps, tabId);
  }
}

export async function launchRewrite(click: MenuClick, tab: chrome.tabs.Tab | undefined, deps: LauncherDeps): Promise<void> {
  const style = parseStyleMenuId(click.menuItemId);
  if (!style || tab?.id === undefined) return;
  const tabId = tab.id;
  const frameId = click.frameId ?? 0;
  const isOwnPage = tab.url?.startsWith(deps.extensionOrigin) ?? false;

  if (!isOwnPage) {
    try {
      await inject(deps, tabId, frameId);
    } catch {
      if (frameId !== 0) await notice(deps, tabId, { code: 'FRAME_INACCESSIBLE' });
      else await flagUnreachable(deps, tabId);
      return;
    }
  }

  if (!(await deps.modelReady())) {
    try {
      await deps.tabs.sendMessage(tabId, { kind: 'echo/notice', error: { code: 'MODEL_NOT_DOWNLOADED' } }, { frameId });
    } catch {
      await flagUnreachable(deps, tabId);
    }
    return;
  }

  // Start loading the model while the page sets up its UI.
  deps.prepareEngine();
  await deps.action.setBadgeText({ tabId, text: '' });

  try {
    // A refusal (no selection, password field, ...) comes back as
    // { ok: false } and is already shown by the page itself.
    await deps.tabs.sendMessage(tabId, { kind: 'echo/start-job', jobId: deps.newJobId(), style }, { frameId });
  } catch {
    // No listener in the frame: it navigated away between click and start.
    await flagUnreachable(deps, tabId);
  }
}
