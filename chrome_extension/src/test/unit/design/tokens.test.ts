/**
 * The design tokens hold together: every theme defines every token, every
 * reference resolves to a colour, and Tailwind gets a class for each.
 * These catch an edit to tokens.json that would otherwise only show up as a
 * missing colour somewhere in the UI.
 */
import { describe, expect, test } from '@jest/globals';
import tokens from '../../../design/tokens.cjs';
import { theme, themeVariables } from '../../../design/tailwind.cjs';

const HEX = /^#[0-9A-F]{6}([0-9A-F]{2})?$/i;

describe('design tokens', () => {
  test('every theme has every token, each a hex colour', () => {
    const names = Object.keys(tokens.themes.light).sort();
    for (const [name, values] of Object.entries(tokens.themes)) {
      expect([name, Object.keys(values).sort()]).toEqual([name, names]);
      for (const [token, value] of Object.entries(values)) expect([name, token, value]).toEqual([name, token, expect.stringMatching(HEX)]);
    }
  });

  test('inverse is dark with mint actions', () => {
    expect(tokens.themes.inverse.surface).toBe(tokens.themes.dark.surface);
    expect(tokens.themes.inverse.accent).toBe(tokens.palette.better['300']);
  });

  test('brand colours and gradients are resolved', () => {
    for (const value of Object.values(tokens.brand)) expect(value).toMatch(HEX);
    expect(tokens.gradient['echo-gradient']).toBe('linear-gradient(135deg, #7650FF 0%, #5B6CFF 45%, #2DD4BF 100%)');
  });

  test('colours become CSS rgb() channels, keeping their own opacity', () => {
    expect(tokens.channels('#7650FF')).toBe('118 80 255');
    expect(tokens.channels('#FFFFFF99')).toBe('255 255 255 / 0.6');
    expect(() => tokens.channels('violet')).toThrow();
  });

  test('Tailwind gets every theme token as a colour, read from its CSS variable', () => {
    const colors = theme.extend.colors as Record<string, unknown>;
    for (const token of Object.keys(tokens.themes.light)) expect(colors[token]).toMatch(new RegExp(`^rgb\\(var\\(--echo-color-${token}\\)`));
    expect(colors['fg']).toBe('rgb(var(--echo-color-fg) / <alpha-value>)');
    expect(colors['success-subtle']).toBe('rgb(var(--echo-color-success-subtle))');
    expect(typeof themeVariables({ selector: ':root', scheme: 'system' })).toBe('object');
  });
});
