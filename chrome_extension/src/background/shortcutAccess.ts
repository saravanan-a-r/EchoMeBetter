/**
 * Keeps the page listener (the shortcuts, and the press and hold that opens
 * the style menu) on websites in step with the user's choices.
 *
 * It runs only while at least one of them is turned on *and* the user has
 * granted the optional site access. Then it is registered as a content
 * script for every page from now on, and injected once into the tabs already
 * open so they do not need a reload. Otherwise it is unregistered.
 */
import { SITE_ACCESS } from '../shared/shortcuts';

export const SHORTCUT_SCRIPT_ID = 'echomebetter-shortcuts';
export const SHORTCUT_SCRIPT = 'shortcuts.js';

export interface ShortcutAccessDeps {
  readonly scripting: Pick<typeof chrome.scripting, 'getRegisteredContentScripts' | 'registerContentScripts' | 'unregisterContentScripts' | 'executeScript'>;
  readonly permissions: Pick<typeof chrome.permissions, 'contains'>;
  readonly tabs: Pick<typeof chrome.tabs, 'query'>;
}

/** Whether something turned on (`enabled`) can work on websites: the user has allowed them. */
export async function activeOnWebsites(enabled: boolean, permissions: ShortcutAccessDeps['permissions']): Promise<boolean> {
  return enabled && (await permissions.contains({ origins: [...SITE_ACCESS.origins] }));
}

export async function syncShortcutScript(active: boolean, deps: ShortcutAccessDeps): Promise<void> {
  const registered = (await deps.scripting.getRegisteredContentScripts({ ids: [SHORTCUT_SCRIPT_ID] })).length > 0;
  if (!active) {
    // Listeners already in open tabs read the setting and stand down on their own.
    if (registered) await deps.scripting.unregisterContentScripts({ ids: [SHORTCUT_SCRIPT_ID] });
    return;
  }
  if (registered) return;
  await deps.scripting.registerContentScripts([
    { id: SHORTCUT_SCRIPT_ID, js: [SHORTCUT_SCRIPT], matches: ['<all_urls>'], allFrames: true, runAt: 'document_start', persistAcrossSessions: true },
  ]);
  const tabs = await deps.tabs.query({});
  await Promise.all(
    tabs.map((tab) =>
      tab.id === undefined
        ? undefined
        : // Pages Chrome keeps extensions out of (its own pages, the Web Store) refuse; nothing to do there.
          deps.scripting.executeScript({ target: { tabId: tab.id, allFrames: true }, files: [SHORTCUT_SCRIPT] }).catch(() => undefined),
    ),
  );
}
