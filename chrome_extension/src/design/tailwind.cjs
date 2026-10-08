/**
 * The design tokens, as Tailwind sees them.
 *
 *   colours     the palette (bg-echo-500, ...) and the theme tokens
 *               (bg-surface, text-fg-muted, ...). A theme token is a CSS
 *               variable, so one class is right in light, dark and inverse
 *               alike, and opacity modifiers still work (bg-accent/20).
 *   variables   `themeVariables()` declares the theme tokens' variables,
 *               where each build needs them (see the two Tailwind configs).
 *   the rest    fonts, shadows, gradients and motion.
 */
const plugin = require('tailwindcss/plugin');
const tokens = require('./tokens.cjs');

const variable = (token) => `--echo-color-${token}`;

/** A theme's tokens as CSS declarations: `--echo-color-fg: 23 18 43`. */
function declarations(themeName) {
  const theme = tokens.themes[themeName];
  return Object.fromEntries(Object.entries(theme).map(([token, hex]) => [variable(token), tokens.channels(hex)]));
}

/** Tokens whose value carries its own opacity in some theme can't take an opacity modifier. */
function hasOwnAlpha(token) {
  return Object.values(tokens.themes).some((theme) => tokens.channels(theme[token]).includes('/'));
}

function themeColors() {
  return Object.fromEntries(
    Object.keys(tokens.themes.light).map((token) => [
      token,
      hasOwnAlpha(token) ? `rgb(var(${variable(token)}))` : `rgb(var(${variable(token)}) / <alpha-value>)`,
    ]),
  );
}

/**
 * Declares the theme variables.
 *   scheme 'system'   `selector` follows the system's light or dark setting.
 *   scheme <theme>    `selector` always uses that theme.
 * Either way, `theme-<name>` classes switch any element and its children to
 * that theme (the welcome page's hero is `theme-inverse`).
 */
function themeVariables({ selector, scheme }) {
  return plugin(({ addBase, addUtilities }) => {
    if (scheme === 'system') {
      addBase({ [selector]: declarations('light') });
      addBase({ '@media (prefers-color-scheme: dark)': { [selector]: declarations('dark') } });
    } else {
      addBase({ [selector]: declarations(scheme) });
    }
    addUtilities(Object.fromEntries(Object.keys(tokens.themes).map((name) => [`.theme-${name}`, declarations(name)])));
  });
}

const easing = (points) => `cubic-bezier(${points.join(', ')})`;
const { duration } = tokens.motion;

const theme = {
  extend: {
    colors: { ...tokens.palette, ...themeColors() },
    fontFamily: { sans: [...tokens.font.sans] },
    backgroundImage: tokens.gradient,
    boxShadow: tokens.shadow,
    keyframes: {
      'echo-wave': {
        '0%, 100%': { opacity: '0.25', transform: 'scale(0.92)' },
        '50%': { opacity: '1', transform: 'scale(1)' },
      },
      'echo-pop-in': {
        '0%': { opacity: '0', transform: 'translateY(6px) scale(0.98)' },
        '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
      },
      'echo-shimmer': {
        '0%': { backgroundPosition: '200% 0' },
        '100%': { backgroundPosition: '-200% 0' },
      },
    },
    animation: {
      'echo-wave': `echo-wave ${duration.pulse} ease-in-out infinite`,
      'echo-pop-in': `echo-pop-in ${duration.enter} ${easing(tokens.motion.easing.enter)} both`,
      'echo-shimmer': `echo-shimmer ${duration.shimmer} linear infinite`,
    },
  },
};

module.exports = { theme, themeVariables };
