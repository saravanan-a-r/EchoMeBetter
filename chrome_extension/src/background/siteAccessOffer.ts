/**
 * The service worker's half of the "hold and shortcuts" offer (the rules are
 * in shared/siteAccessOffer.ts): decide after a Replace whether to offer the
 * site access, and ask for it when the user presses Allow.
 */
import type { AccessOfferReply, AccessResult } from '../shared/messages';
import type { SettingsFocus } from '../shared/popupIntent';
import { wantsWebsites, type Settings } from '../shared/settings';
import { SITE_ACCESS, type KeyPlatform } from '../shared/shortcuts';
import { countReplace, parseOfferRecord, SITE_ACCESS_OFFER_KEY } from '../shared/siteAccessOffer';

export interface SiteAccessOfferDeps {
  readonly storage: Pick<chrome.storage.StorageArea, 'get' | 'set'>;
  readonly hasAccess: () => Promise<boolean>;
  readonly settings: () => Promise<Pick<Settings, 'shortcutsEnabled' | 'holdMenuEnabled'>>;
  readonly platform: () => Promise<KeyPlatform>;
}

/** `ownPage`: the Replace happened on one of the extension's own pages, which have their own switch for this. */
export async function offerAfterReplace(ownPage: boolean, deps: SiteAccessOfferDeps): Promise<AccessOfferReply> {
  const stored = await deps.storage.get(SITE_ACCESS_OFFER_KEY);
  const settings = await deps.settings();
  const eligible = !ownPage && wantsWebsites(settings) && !(await deps.hasAccess());
  const { record, offer } = countReplace(parseOfferRecord(stored[SITE_ACCESS_OFFER_KEY]), eligible);
  await deps.storage.set({ [SITE_ACCESS_OFFER_KEY]: record });
  if (!offer) return { offer: false };
  return { offer: true, hold: settings.holdMenuEnabled, shortcuts: settings.shortcutsEnabled, platform: await deps.platform() };
}

export interface RequestAccessDeps {
  readonly request: (permissions: { origins: string[] }) => Promise<boolean>;
  /** Open the popup on the setting that holds the Allow button. */
  readonly openSettings: (focus: SettingsFocus) => Promise<void>;
}

/**
 * Chrome shows its permission prompt only for a user's click. The click on
 * the page's card is passed on to the service worker with the message, which
 * Chrome accepts; where it does not, the popup (which has the same button)
 * opens instead, so Allow always leads somewhere.
 */
export async function requestSiteAccess(deps: RequestAccessDeps): Promise<AccessResult> {
  try {
    return (await deps.request({ origins: [...SITE_ACCESS.origins] })) ? 'granted' : 'denied';
  } catch {
    await deps.openSettings('site-access');
    return 'opened';
  }
}
