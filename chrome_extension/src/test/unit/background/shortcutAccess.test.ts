import { describe, expect, jest, test } from '@jest/globals';
import { SHORTCUT_SCRIPT, SHORTCUT_SCRIPT_ID, shortcutsActive, syncShortcutScript, type ShortcutAccessDeps } from '../../../background/shortcutAccess';

function deps(options: { registered?: boolean; granted?: boolean; failingTab?: number } = {}) {
  const value = {
    scripting: {
      getRegisteredContentScripts: jest.fn(async () => (options.registered ? [{ id: SHORTCUT_SCRIPT_ID }] : [])),
      registerContentScripts: jest.fn(async (_scripts: unknown[]) => undefined),
      unregisterContentScripts: jest.fn(async (_filter: { ids: string[] }) => undefined),
      executeScript: jest.fn(async ({ target }: { target: { tabId: number; allFrames?: boolean }; files: string[] }) => {
        if (target.tabId === options.failingTab) throw new Error('Cannot access a chrome:// URL');
        return [];
      }),
    },
    permissions: { contains: jest.fn(async () => options.granted ?? true) },
    tabs: { query: jest.fn(async () => [{ id: 1 }, { id: 2 }, {}]) },
  };
  return value as typeof value & ShortcutAccessDeps;
}

describe('shortcut access', () => {
  test('active only when turned on and websites are allowed', async () => {
    expect(await shortcutsActive(true, deps({ granted: true }).permissions)).toBe(true);
    expect(await shortcutsActive(true, deps({ granted: false }).permissions)).toBe(false);
    const off = deps();
    expect(await shortcutsActive(false, off.permissions)).toBe(false);
    expect(off.permissions.contains).not.toHaveBeenCalled();
  });

  test('turning on registers the listener everywhere and reaches tabs already open', async () => {
    const value = deps({ failingTab: 2 });
    await syncShortcutScript(true, value);
    expect(value.scripting.registerContentScripts).toHaveBeenCalledWith([
      expect.objectContaining({ id: SHORTCUT_SCRIPT_ID, js: [SHORTCUT_SCRIPT], matches: ['<all_urls>'], allFrames: true, persistAcrossSessions: true }),
    ]);
    expect(value.scripting.executeScript).toHaveBeenCalledTimes(2);
    expect(value.scripting.executeScript).toHaveBeenCalledWith({ target: { tabId: 1, allFrames: true }, files: [SHORTCUT_SCRIPT] });
  });

  test('already registered: nothing to do', async () => {
    const value = deps({ registered: true });
    await syncShortcutScript(true, value);
    expect(value.scripting.registerContentScripts).not.toHaveBeenCalled();
    expect(value.scripting.executeScript).not.toHaveBeenCalled();
  });

  test('turning off unregisters it', async () => {
    const value = deps({ registered: true });
    await syncShortcutScript(false, value);
    expect(value.scripting.unregisterContentScripts).toHaveBeenCalledWith({ ids: [SHORTCUT_SCRIPT_ID] });
    const never = deps({ registered: false });
    await syncShortcutScript(false, never);
    expect(never.scripting.unregisterContentScripts).not.toHaveBeenCalled();
  });
});
