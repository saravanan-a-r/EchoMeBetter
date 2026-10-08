/** Extension pages (popup, welcome): full Tailwind with its base reset, in the system's light or dark theme. */
const { themeVariables } = require('./src/design/tailwind.cjs');

module.exports = {
  presets: [require('./tailwind.preset.cjs')],
  content: ['./src/ui/**/*.{ts,tsx,html}'],
  plugins: [themeVariables({ selector: ':root', scheme: 'system' })],
};
