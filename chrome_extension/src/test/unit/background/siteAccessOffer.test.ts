import { describe, expect, jest, test } from '@jest/globals';
import { offerAfterReplace, requestSiteAccess, type SiteAccessOfferDeps } from '../../../background/siteAccessOffer';
import type { SettingsFocus } from '../../../shared/popupIntent';
import { SITE_ACCESS_OFFER_KEY } from '../../../shared/siteAccessOffer';

function deps(overrides: Partial<{ hasAccess: boolean; shortcutsEnabled: boolean; holdMenuEnabled: boolean }> = {}) {
  const stored: Record<string, unknown> = {};
  const value: SiteAccessOfferDeps = {
    storage: {
      get: (async (key: string) => (key in stored ? { [key]: stored[key] } : {})) as SiteAccessOfferDeps['storage']['get'],
      set: (async (items: Record<string, unknown>) => void Object.assign(stored, items)) as SiteAccessOfferDeps['storage']['set'],
    },
    hasAccess: async () => overrides.hasAccess ?? false,
    settings: async () => ({ shortcutsEnabled: overrides.shortcutsEnabled ?? true, holdMenuEnabled: overrides.holdMenuEnabled ?? true }),
    platform: async () => 'mac',
  };
  return { value, stored };
}

describe('offerAfterReplace', () => {
  test('offers on the second Replace, with the features and keys to show', async () => {
    const { value, stored } = deps();
    expect(await offerAfterReplace(false, value)).toEqual({ offer: false });
    expect(await offerAfterReplace(false, value)).toEqual({ offer: true, hold: true, shortcuts: true, platform: 'mac' });
    expect(stored[SITE_ACCESS_OFFER_KEY]).toEqual({ replaces: 2, offered: 1 });
    expect(await offerAfterReplace(false, value)).toEqual({ offer: false });
  });

  test('only offers what the user kept switched on', async () => {
    const { value } = deps({ holdMenuEnabled: false });
    await offerAfterReplace(false, value);
    expect(await offerAfterReplace(false, value)).toMatchObject({ offer: true, hold: false, shortcuts: true });
  });

  test('stays quiet when the access is already there, nothing needs it, or the Replace was on an extension page', async () => {
    for (const quiet of [deps({ hasAccess: true }), deps({ shortcutsEnabled: false, holdMenuEnabled: false })]) {
      await offerAfterReplace(false, quiet.value);
      expect(await offerAfterReplace(false, quiet.value)).toEqual({ offer: false });
    }
    const page = deps();
    await offerAfterReplace(true, page.value);
    expect(await offerAfterReplace(true, page.value)).toEqual({ offer: false });
    expect(await offerAfterReplace(false, page.value)).toMatchObject({ offer: true });
  });
});

describe('requestSiteAccess', () => {
  test('reports what the user answered', async () => {
    const openSettings = jest.fn(async (_focus: SettingsFocus) => undefined);
    expect(await requestSiteAccess({ request: async () => true, openSettings })).toBe('granted');
    expect(await requestSiteAccess({ request: async () => false, openSettings })).toBe('denied');
    expect(openSettings).not.toHaveBeenCalled();
  });

  test('where Chrome will not show the prompt from here, opens the popup on the button that does', async () => {
    const openSettings = jest.fn(async (_focus: SettingsFocus) => undefined);
    const request = async () => {
      throw new Error('This function must be called during a user gesture');
    };
    expect(await requestSiteAccess({ request, openSettings })).toBe('opened');
    expect(openSettings).toHaveBeenCalledWith('site-access');
  });
});
