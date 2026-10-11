/**
 * The overlay's stylesheet must not depend on the page it is shown on.
 *
 * `rem` is the font size of the page's <html>; a site with `html { font-size:
 * 10px }` shrank the overlay's paddings and line heights to 62% while its
 * text stayed put, and the style menu overlapped itself. The overlay is
 * built in pixels, which are the same everywhere.
 */
import { describe, expect, test } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import postcss from 'postcss';
import tailwindcss from 'tailwindcss';
import remToPx from '../../../../tooling/remToPx.cjs';

const root = join(__dirname, '../../../..');

async function compile(css: string, withPlugin = true) {
  const plugins = [tailwindcss({ config: join(root, 'tailwind.foreground.config.cjs') }), ...(withPlugin ? [remToPx()] : [])];
  return (await postcss(plugins).process(css, { from: join(root, 'src/foreground/ui/overlay.css') })).css;
}

describe('rem to px', () => {
  test('converts declarations at 16px to the rem, exactly, and leaves media queries and other units alone', async () => {
    const out = await postcss([remToPx()]).process(
      `.a { padding: 0.375rem 1.75rem; margin: -0.125rem; width: calc(100% - 2.5rem); line-height: 1.5; height: 1em; border-radius: 9999px; box-shadow: 0 1rem 3rem #0004 }
       @media (min-width: 40rem) { .b { gap: 0.5rem } }`,
      { from: undefined },
    );
    expect(out.css).toContain('padding: 6px 28px');
    expect(out.css).toContain('margin: -2px');
    expect(out.css).toContain('width: calc(100% - 40px)');
    expect(out.css).toContain('line-height: 1.5; height: 1em; border-radius: 9999px');
    expect(out.css).toContain('box-shadow: 0 16px 48px #0004');
    expect(out.css).toContain('@media (min-width: 40rem)');
    expect(out.css).toContain('gap: 8px');
  });
});

describe('the overlay stylesheet', () => {
  const overlay = readFileSync(join(root, 'src/foreground/ui/overlay.css'), 'utf8');

  test('is in pixels once built: nothing in it follows the page\'s root font size', async () => {
    const css = await compile(overlay);
    expect(css).not.toMatch(/\d\s*rem\b/);
    // Spot-check what the cards are laid out with: icon box, line height, radius.
    // Tailwind writes the prefixed class `_echo_$_h-7` as `._echo_\$_h-7`.
    const rule = (name: string) => css.match(new RegExp(String.raw`\._echo_\\\$_${name} \{[^}]*\}`))?.[0] ?? '(missing)';
    expect(rule('h-7')).toContain('height: 28px');
    expect(rule('leading-5')).toContain('line-height: 20px');
    expect(rule('rounded-2xl')).toContain('border-radius: 16px');
  });

  test('would not be, without the conversion (the test would otherwise pass for nothing)', async () => {
    expect(await compile(overlay, false)).toMatch(/\d\s*rem\b/);
  });
});
