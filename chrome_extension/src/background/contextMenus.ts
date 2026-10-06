/**
 * The right-click entry point: "EchoMeBetter ▸ Professional / Grammar / ...",
 * shown only on editable fields (inputs, textareas, contenteditable).
 * While keyboard shortcuts are on, each style shows its shortcut.
 */
import { isStyleId, STYLES, type StyleId } from '../shared/styles';

export const ROOT_MENU_ID = 'echomebetter';
const STYLE_PREFIX = `${ROOT_MENU_ID}:`;

export function styleMenuId(style: StyleId): string {
  return `${STYLE_PREFIX}${style}`;
}

export function parseStyleMenuId(menuItemId: string | number): StyleId | null {
  if (typeof menuItemId !== 'string' || !menuItemId.startsWith(STYLE_PREFIX)) return null;
  const style = menuItemId.slice(STYLE_PREFIX.length);
  return isStyleId(style) ? style : null;
}

type MenusApi = Pick<typeof chrome.contextMenus, 'create' | 'removeAll'>;

function create(menus: MenusApi, properties: chrome.contextMenus.CreateProperties): Promise<void> {
  return new Promise((resolve, reject) => {
    menus.create(properties, () => {
      const error = chrome.runtime.lastError;
      if (error) reject(new Error(error.message));
      else resolve();
    });
  });
}

/** Menus have no column for shortcuts, so the shortcut follows the label. */
export function styleMenuTitle(label: string, shortcut: string | null): string {
  return shortcut ? `${label}   (${shortcut})` : label;
}

export async function registerContextMenus(menus: MenusApi = chrome.contextMenus, shortcutFor: (style: StyleId) => string | null = () => null): Promise<void> {
  await menus.removeAll();
  await create(menus, { id: ROOT_MENU_ID, title: 'EchoMeBetter', contexts: ['editable'] });
  for (const style of STYLES) {
    const title = styleMenuTitle(style.label, shortcutFor(style.id));
    await create(menus, { id: styleMenuId(style.id), parentId: ROOT_MENU_ID, title, contexts: ['editable'] });
  }
}
