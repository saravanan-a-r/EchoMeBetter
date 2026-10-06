import { describe, expect, test } from '@jest/globals';
import { detectPlatform, matchShortcut, platformFromOs, SHORTCUT_LETTERS, shortcutKeys, shortcutLabel, spokenShortcut, type KeyChord } from '../../../shared/shortcuts';
import { STYLE_IDS } from '../../../shared/styles';

const chord = (key: string, modifiers: Partial<KeyChord> = {}): KeyChord => ({
  key,
  code: `Key${key.toUpperCase()}`,
  altKey: false,
  ctrlKey: false,
  shiftKey: false,
  metaKey: false,
  ...modifiers,
});

describe('shortcuts', () => {
  test('every style has its own letter', () => {
    const letters = STYLE_IDS.map((style) => SHORTCUT_LETTERS[style]);
    expect(new Set(letters).size).toBe(STYLE_IDS.length);
  });

  test('Windows, Linux and ChromeOS use Alt+Shift+letter', () => {
    expect(matchShortcut(chord('P', { altKey: true, shiftKey: true }), 'other')).toBe('professional');
    expect(matchShortcut(chord('E', { altKey: true, shiftKey: true }), 'other')).toBe('elaborate');
    // Extra or missing modifiers are someone else's shortcut.
    expect(matchShortcut(chord('P', { altKey: true }), 'other')).toBeNull();
    expect(matchShortcut(chord('P', { altKey: true, shiftKey: true, ctrlKey: true }), 'other')).toBeNull();
    expect(matchShortcut(chord('P', { ctrlKey: true, shiftKey: true }), 'other')).toBeNull();
    expect(matchShortcut(chord('X', { altKey: true, shiftKey: true }), 'other')).toBeNull();
  });

  test('macOS uses Control+Shift+letter, never Command or Option', () => {
    expect(matchShortcut(chord('G', { ctrlKey: true, shiftKey: true }), 'mac')).toBe('grammar');
    expect(matchShortcut(chord('G', { metaKey: true, shiftKey: true }), 'mac')).toBeNull();
    expect(matchShortcut(chord('G', { altKey: true, shiftKey: true }), 'mac')).toBeNull();
  });

  test('the typed letter follows the keyboard layout; the physical key is the fallback', () => {
    // AZERTY: the key in the QWERTY "Q" position types "a".
    expect(matchShortcut({ ...chord('c', { altKey: true, shiftKey: true }), code: 'KeyX' }, 'other')).toBe('concise');
    // Option turns letters into symbols on some layouts.
    expect(matchShortcut({ ...chord('ƒ', { altKey: true, shiftKey: true }), code: 'KeyF' }, 'other')).toBe('friendly');
  });

  test('written the way each platform writes shortcuts', () => {
    expect(shortcutLabel('professional', 'mac')).toBe('⌃⇧P');
    expect(shortcutLabel('professional', 'other')).toBe('Alt+Shift+P');
    expect(shortcutKeys('concise', 'other')).toEqual(['Alt', 'Shift', 'C']);
    expect(spokenShortcut('concise', 'mac')).toBe('Control Shift C');
  });

  test('platform detection', () => {
    expect(detectPlatform({ userAgentData: { platform: 'macOS' } })).toBe('mac');
    expect(detectPlatform({ platform: 'MacIntel' })).toBe('mac');
    expect(detectPlatform({ userAgentData: { platform: 'Windows' }, platform: 'Win32' })).toBe('other');
    expect(detectPlatform({ platform: 'Linux x86_64' })).toBe('other');
    expect(detectPlatform({})).toBe('other');
    expect(platformFromOs('mac')).toBe('mac');
    expect(platformFromOs('cros')).toBe('other');
  });
});
