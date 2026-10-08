/**
 * The in-page overlay. It lives in a shadow root, so the base reset is off:
 * the overlay sets its own baseline on :host instead of resetting a page
 * that is not ours. It always uses the inverse theme: a dark surface that
 * reads the same on any website, light or dark.
 */
const { themeVariables } = require('./src/design/tailwind.cjs');

module.exports = {
  presets: [require('./tailwind.preset.cjs')],
  content: ['./src/foreground/**/*.{ts,tsx}', './src/ui/brand/**/*.{ts,tsx}'],
  corePlugins: { preflight: false },
  plugins: [themeVariables({ selector: '.echomebetter-root', scheme: 'inverse' })],
};
