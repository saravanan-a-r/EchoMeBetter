/**
 * When to offer the site access that press and hold and the keyboard
 * shortcuts need.
 *
 * Right-click works without it, so the offer waits until the user has seen
 * EchoMeBetter work: it comes right after their second Replace. A user who
 * says "Not now" is asked once more, ten Replaces later, and then left alone;
 * the popup and the welcome page keep the switch for good.
 *
 * The count lives in `chrome.storage.local`, so it follows the user across
 * tabs and browser restarts.
 */

export const SITE_ACCESS_OFFER_KEY = 'siteAccessOffer';

/** The Replace (counted from the first) at which the first, and then the one more, offer may come. */
export const OFFER_AT_REPLACE = [2, 12] as const;

export interface OfferRecord {
  /** Rewrites the user has put in place. */
  readonly replaces: number;
  /** Times the offer has been shown. */
  readonly offered: number;
}

export const NO_OFFERS_YET: OfferRecord = { replaces: 0, offered: 0 };

const count = (value: unknown): number => (typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : 0);

export function parseOfferRecord(raw: unknown): OfferRecord {
  if (typeof raw !== 'object' || raw === null) return NO_OFFERS_YET;
  const stored = raw as Partial<Record<keyof OfferRecord, unknown>>;
  return { replaces: count(stored.replaces), offered: count(stored.offered) };
}

/**
 * One more Replace. `eligible` is whether an offer would make sense here at all
 * (the user wants hold or shortcuts, does not have the access yet, and is on a
 * website rather than on the welcome page, where the switch is on the page).
 */
export function countReplace(record: OfferRecord, eligible: boolean): { readonly record: OfferRecord; readonly offer: boolean } {
  const replaces = record.replaces + 1;
  const due = OFFER_AT_REPLACE[record.offered];
  const offer = eligible && due !== undefined && replaces >= due;
  return { record: { replaces, offered: offer ? record.offered + 1 : record.offered }, offer };
}
