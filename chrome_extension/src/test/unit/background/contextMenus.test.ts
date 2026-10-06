import { describe, expect, jest, test } from '@jest/globals';
import { parseStyleMenuId, registerContextMenus, ROOT_MENU_ID, styleMenuId, styleMenuTitle } from '../../../background/contextMenus';
import { STYLES } from '../../../shared/styles';

describe('context menu', () => {
  test('registers "EchoMeBetter" with one child per style, on editable fields only, in catalogue order', async () => {
    (globalThis as { chrome?: unknown }).chrome = { runtime: { lastError: undefined } };
    const created: chrome.contextMenus.CreateProperties[] = [];
    const menus = {
      removeAll: jest.fn(async () => undefined),
      create: jest.fn((properties: chrome.contextMenus.CreateProperties, done?: () => void) => {
        created.push(properties);
        done?.();
        return properties.id!;
      }),
    };
    await registerContextMenus(menus as never);
    expect(menus.removeAll).toHaveBeenCalledTimes(1);
    expect(created[0]).toEqual({ id: ROOT_MENU_ID, title: 'EchoMeBetter', contexts: ['editable'] });
    expect(created.slice(1).map((item) => [item.title, item.parentId, item.contexts])).toEqual(
      STYLES.map((style) => [style.label, ROOT_MENU_ID, ['editable']]),
    );
  });

  test('with shortcuts on, each style shows its shortcut', async () => {
    (globalThis as { chrome?: unknown }).chrome = { runtime: { lastError: undefined } };
    const created: chrome.contextMenus.CreateProperties[] = [];
    const menus = {
      removeAll: jest.fn(async () => undefined),
      create: jest.fn((properties: chrome.contextMenus.CreateProperties, done?: () => void) => {
        created.push(properties);
        done?.();
        return properties.id!;
      }),
    };
    await registerContextMenus(menus as never, (style) => (style === 'grammar' ? '⌃⇧G' : null));
    expect(created.find((item) => item.id === styleMenuId('grammar'))!.title).toBe('Grammar   (⌃⇧G)');
    expect(created.find((item) => item.id === styleMenuId('concise'))!.title).toBe('Concise');
    expect(styleMenuTitle('Friendly', 'Alt+Shift+F')).toBe('Friendly   (Alt+Shift+F)');
  });

  test('menu ids round-trip to styles; anything else is not a style', () => {
    for (const style of STYLES) expect(parseStyleMenuId(styleMenuId(style.id))).toBe(style.id);
    expect(parseStyleMenuId(ROOT_MENU_ID)).toBeNull();
    expect(parseStyleMenuId('echomebetter:shouting')).toBeNull();
    expect(parseStyleMenuId(42)).toBeNull();
  });
});
