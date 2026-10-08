/**
 * Where the toolbar popup should open, when something other than the
 * toolbar button opens it.
 *
 * The popup always loads the same page, so the request is left in
 * `chrome.storage.session` just before it opens, and the popup takes it
 * (reading it removes it) as it starts. A request that is never taken (the
 * popup could not open) goes stale, so the next time the user opens the
 * popup themselves it starts on its home page as usual.
 */

/** A setting the popup scrolls to and highlights. */
export const SETTINGS_FOCUSES = ['keep-awake'] as const;

export type SettingsFocus = (typeof SETTINGS_FOCUSES)[number];

export interface PopupIntent {
  readonly page: 'settings';
  readonly focus: SettingsFocus;
  /** Date.now() when it was asked for. */
  readonly at: number;
}

export const POPUP_INTENT_KEY = 'popupIntent';

/** Long enough for the popup to open, short enough not to surprise the user later. */
export const POPUP_INTENT_TTL_MS = 30_000;

export function isSettingsFocus(value: unknown): value is SettingsFocus {
  return typeof value === 'string' && (SETTINGS_FOCUSES as readonly string[]).includes(value);
}

export function parsePopupIntent(value: unknown, now: number): PopupIntent | null {
  if (typeof value !== 'object' || value === null) return null;
  const intent = value as Partial<Record<keyof PopupIntent, unknown>>;
  if (intent.page !== 'settings' || !isSettingsFocus(intent.focus) || typeof intent.at !== 'number') return null;
  const age = now - intent.at;
  return age >= 0 && age <= POPUP_INTENT_TTL_MS ? { page: intent.page, focus: intent.focus, at: intent.at } : null;
}

export async function leavePopupIntent(focus: SettingsFocus, now = Date.now(), storage: chrome.storage.StorageArea = chrome.storage.session): Promise<void> {
  await storage.set({ [POPUP_INTENT_KEY]: { page: 'settings', focus, at: now } satisfies PopupIntent });
}

/** The waiting request, if there is a fresh one; it is removed either way. */
export async function takePopupIntent(now = Date.now(), storage: chrome.storage.StorageArea = chrome.storage.session): Promise<PopupIntent | null> {
  const stored = await storage.get(POPUP_INTENT_KEY);
  if (!(POPUP_INTENT_KEY in stored)) return null;
  await storage.remove(POPUP_INTENT_KEY);
  return parsePopupIntent(stored[POPUP_INTENT_KEY], now);
}
