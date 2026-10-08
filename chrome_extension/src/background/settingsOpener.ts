/**
 * Opens the toolbar popup on one of its settings, for a button on a web page
 * (a toast's "keep it awake longer").
 *
 * The popup is asked where to start through a popup intent (see
 * shared/popupIntent.ts), then opened. `chrome.action.openPopup` needs Chrome
 * 127 and a focused browser window; where it can't open the popup, the same
 * page opens in a small window of its own, so the button always leads
 * somewhere.
 */
import type { SettingsFocus } from '../shared/popupIntent';

export const POPUP_PAGE = 'ui/popup/popup.html';

/** Sized for the popup's 360px-wide page. */
export const POPUP_WINDOW = { width: 380, height: 640 } as const;

export interface SettingsOpenerDeps {
  readonly leaveIntent: (focus: SettingsFocus) => Promise<void>;
  /** chrome.action.openPopup, where the browser has it. */
  readonly openPopup?: (options: { windowId?: number }) => Promise<void>;
  readonly createWindow: (options: chrome.windows.CreateData) => Promise<unknown>;
  readonly popupUrl: string;
}

export async function openSettings(focus: SettingsFocus, windowId: number | undefined, deps: SettingsOpenerDeps): Promise<void> {
  await deps.leaveIntent(focus);
  try {
    if (!deps.openPopup) throw new Error('chrome.action.openPopup is not available');
    await deps.openPopup(windowId === undefined ? {} : { windowId });
  } catch {
    await deps.createWindow({ url: deps.popupUrl, type: 'popup', focused: true, ...POPUP_WINDOW });
  }
}
