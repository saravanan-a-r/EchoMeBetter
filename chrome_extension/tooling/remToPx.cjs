/**
 * PostCSS plugin: every `rem` in a declaration becomes the pixels it is at
 * the browser's default root size (1rem = 16px).
 *
 * For the in-page overlay's stylesheet only. `rem` means the font size of
 * the *page's* <html>, and the overlay sits inside someone else's page: a site
 * with `html { font-size: 62.5% }` would shrink every rem-based padding, gap,
 * radius and line height to 62%, while the pixel-sized text stays the same,
 * and the overlay would overlap itself. A shadow root cannot isolate `rem`;
 * pixels are the same on every site. The extension's own pages keep `rem`:
 * they are our document, with the root size we expect.
 *
 * Only declarations change. A `rem` in a media query is the browser's own
 * default size, not the page's, so it is left alone.
 */
const REM = /(-?\d*\.?\d+)rem\b/g;
const PIXELS_PER_REM = 16;

const toPixels = (value) =>
  value.replace(REM, (_match, rem) => {
    // Four places are exact for every Tailwind step, and keep the CSS free of 0.30000000000000004px.
    const pixels = Math.round(Number(rem) * PIXELS_PER_REM * 10000) / 10000;
    return `${pixels}px`;
  });

/** @type {import('postcss').PluginCreator<void>} */
const remToPx = () => ({
  postcssPlugin: 'echomebetter-rem-to-px',
  Declaration(declaration) {
    declaration.value = toPixels(declaration.value);
  },
});
remToPx.postcss = true;

module.exports = remToPx;
