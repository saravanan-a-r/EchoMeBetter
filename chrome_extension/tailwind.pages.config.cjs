/** Extension pages (popup, welcome): full Tailwind with its base reset. */
module.exports = {
  presets: [require('./tailwind.preset.cjs')],
  content: ['./src/ui/**/*.{ts,tsx,html}'],
};
