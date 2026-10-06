/**
 * Keyboard shortcuts: one per style, the same letter on every platform.
 *
 *   Windows, Linux, ChromeOS   Alt+Shift+<letter>
 *   macOS                      Control+Shift+<letter>  (⌃⇧)
 *
 * Both chords stay clear of the browser's own shortcuts (Ctrl/⌘ and
 * Ctrl/⌘+Shift) and are rarely used by web apps. They are also only taken
 * when text is selected in an editable field, so a site that uses the same
 * keys for something else still gets them everywhere else.
 *
 * Listening on web pages needs access to them, an optional permission the
 * user grants from the popup; nothing runs on pages until then.
 */
import { STYLES, type StyleId } from './styles';

export type KeyPlatform = 'mac' | 'other';

/** The optional host permission that lets the shortcut listener run on websites. */
export const SITE_ACCESS = { origins: ['<all_urls>'] } as const;

export const SHORTCUT_LETTERS: Readonly<Record<StyleId, string>> = {
  professional: 'P',
  grammar: 'G',
  friendly: 'F',
  concise: 'C',
  elaborate: 'E',
};

interface NavigatorLike {
  readonly platform?: string;
  readonly userAgentData?: { readonly platform?: string };
}

export function detectPlatform(nav: NavigatorLike): KeyPlatform {
  const name = nav.userAgentData?.platform || nav.platform || '';
  return /mac/i.test(name) ? 'mac' : 'other';
}

/** chrome.runtime.getPlatformInfo().os, for the service worker. */
export function platformFromOs(os: string): KeyPlatform {
  return os === 'mac' ? 'mac' : 'other';
}

/** The keys to show, one per <kbd>. */
export function shortcutKeys(style: StyleId, platform: KeyPlatform): readonly string[] {
  const letter = SHORTCUT_LETTERS[style];
  return platform === 'mac' ? ['⌃', '⇧', letter] : ['Alt', 'Shift', letter];
}

/** As one string, the way each platform writes shortcuts in its menus. */
export function shortcutLabel(style: StyleId, platform: KeyPlatform): string {
  return platform === 'mac' ? shortcutKeys(style, platform).join('') : shortcutKeys(style, platform).join('+');
}

/** For screen readers, which do not read ⌃ and ⇧ well. */
export function spokenShortcut(style: StyleId, platform: KeyPlatform): string {
  return `${platform === 'mac' ? 'Control' : 'Alt'} Shift ${SHORTCUT_LETTERS[style]}`;
}

export interface KeyChord {
  readonly key: string;
  readonly code: string;
  readonly altKey: boolean;
  readonly ctrlKey: boolean;
  readonly shiftKey: boolean;
  readonly metaKey: boolean;
}

function chordLetter(chord: KeyChord): string | null {
  // The typed letter follows the user's keyboard layout; Alt can turn it into
  // another character on some layouts, and then the physical key decides.
  if (/^[a-z]$/i.test(chord.key)) return chord.key.toUpperCase();
  const physical = /^Key([A-Z])$/.exec(chord.code);
  return physical ? physical[1]! : null;
}

export function matchShortcut(chord: KeyChord, platform: KeyPlatform): StyleId | null {
  const modifiers =
    platform === 'mac'
      ? chord.ctrlKey && chord.shiftKey && !chord.altKey && !chord.metaKey
      : chord.altKey && chord.shiftKey && !chord.ctrlKey && !chord.metaKey;
  if (!modifiers) return null;
  const letter = chordLetter(chord);
  return STYLES.find((style) => SHORTCUT_LETTERS[style.id] === letter)?.id ?? null;
}
