/**
 * The in-page overlay. It lives in a shadow root, so the base reset is off:
 * the overlay sets its own baseline on :host instead of resetting a page
 * that is not ours.
 */
module.exports = {
  presets: [require('./tailwind.preset.cjs')],
  content: ['./src/foreground/**/*.{ts,tsx}', './src/ui/brand/**/*.{ts,tsx}'],
  corePlugins: { preflight: false },
};
