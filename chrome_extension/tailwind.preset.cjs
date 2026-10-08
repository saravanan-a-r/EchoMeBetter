/**
 * Shared by every Tailwind build (extension pages and the in-page overlay):
 * the design tokens (src/design) as Tailwind's theme. Classes are prefixed
 * `_echo_$_` so they can never collide with a host page's own utility classes.
 */
const { theme } = require('./src/design/tailwind.cjs');

module.exports = {
  prefix: '_echo_$_',
  darkMode: 'media',
  theme,
  plugins: [],
};
