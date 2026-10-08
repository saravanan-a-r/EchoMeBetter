import { describe, expect, jest, test } from '@jest/globals';
import { openSettings, POPUP_WINDOW, type SettingsOpenerDeps } from '../../../background/settingsOpener';
import type { SettingsFocus } from '../../../shared/popupIntent';

function deps(openPopup?: SettingsOpenerDeps['openPopup']) {
  const order: string[] = [];
  return {
    order,
    leaveIntent: jest.fn(async (focus: SettingsFocus) => void order.push(`intent ${focus}`)),
    openPopup: openPopup && jest.fn(async (options: { windowId?: number }) => {
      order.push('popup');
      await openPopup(options);
    }),
    createWindow: jest.fn(async (_options: chrome.windows.CreateData) => void order.push('window')),
    popupUrl: 'chrome-extension://id/ui/popup/popup.html',
  };
}

describe('openSettings', () => {
  test('leaves the intent for the popup, then opens the popup over the window the button was in', async () => {
    const opener = deps(async () => undefined);
    await openSettings('keep-awake', 7, opener);
    expect(opener.order).toEqual(['intent keep-awake', 'popup']);
    expect(opener.openPopup).toHaveBeenCalledWith({ windowId: 7 });
    expect(opener.createWindow).not.toHaveBeenCalled();
  });

  test('where the popup cannot open, the same page opens in a small window', async () => {
    for (const opener of [deps(async () => Promise.reject(new Error('no active window'))), deps(undefined)]) {
      await openSettings('keep-awake', undefined, opener);
      expect(opener.order.at(-1)).toBe('window');
      expect(opener.createWindow).toHaveBeenCalledWith({ url: opener.popupUrl, type: 'popup', focused: true, ...POPUP_WINDOW });
    }
  });
});
