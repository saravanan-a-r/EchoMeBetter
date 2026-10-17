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
      'echo-ripple': {
        '0%': { opacity: '0', transform: 'scale(0.96)' },
        '12%': { opacity: '0.5' },
        '100%': { opacity: '0', transform: 'scale(1.7)' },
      },
      'echo-glow-breath': {
        '0%, 100%': { opacity: '0.55', transform: 'scale(0.95)' },
        '50%': { opacity: '0.9', transform: 'scale(1.05)' },
      },
      'echo-press': {
        '0%': { opacity: '0.9', transform: 'scale(0.4)' },
        '100%': { opacity: '0', transform: 'scale(1.4)' },
      },
      'echo-hold-fill': {
        '100%': { strokeDashoffset: '0' },
      },
      'echo-flash': {
        '0%': { backgroundColor: 'rgba(45, 212, 191, 0.3)' },
        '100%': { backgroundColor: 'transparent' },
      },
      'echo-demo-stage': {
        '0%': { opacity: '0' },
        '5%, 90%': { opacity: '1' },
        '97%, 100%': { opacity: '0' },
      },
      'echo-demo-select': {
        '0%, 8%': { transform: 'scaleX(0)' },
        '18%, 100%': { transform: 'scaleX(1)' },
      },
      'echo-demo-pointer': {
        '0%, 12%': { opacity: '0', transform: 'translate(-18px, 14px)' },
        '20%, 56%': { opacity: '1', transform: 'translate(0, 0)' },
        '66%, 100%': { opacity: '1', transform: 'translate(46px, 34px)' },
      },
      // The ring fills over a fifth of the loop: one second, as long as the real hold.
      'echo-demo-ring': {
        '0%, 20%': { opacity: '0', strokeDashoffset: '69.115' },
        '21%': { opacity: '1', strokeDashoffset: '69.115' },
        '41%': { opacity: '1', strokeDashoffset: '0' },
        '46%, 100%': { opacity: '0', strokeDashoffset: '0' },
      },
      'echo-demo-menu': {
        '0%, 44%': { opacity: '0', transform: 'translateY(4px) scale(0.96)' },
        '52%, 100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
      },
      'echo-demo-pick': {
        '0%, 64%': { opacity: '0' },
        '70%, 100%': { opacity: '1' },
      },
      'echo-demo-key-1': {
        '0%, 10%': { opacity: '0' },
        '14%, 100%': { opacity: '1' },
      },
      'echo-demo-key-2': {
        '0%, 18%': { opacity: '0' },
        '22%, 100%': { opacity: '1' },
      },
      'echo-demo-key-3': {
        '0%, 26%': { opacity: '0' },
        '30%, 100%': { opacity: '1' },
      },
      'echo-demo-chip': {
        '0%, 40%': { opacity: '0', transform: 'translateY(6px)' },
        '48%, 100%': { opacity: '1', transform: 'translateY(0)' },
      },
      'echo-shimmer': {
        '0%': { backgroundPosition: '200% 0' },
        '100%': { backgroundPosition: '-200% 0' },
      },
    },
    animation: {
      'echo-wave': `echo-wave ${duration.pulse} ease-in-out infinite`,
      'echo-pop-in': `echo-pop-in ${duration.enter} ${easing(tokens.motion.easing.enter)} both`,
      'echo-ripple': `echo-ripple ${duration.ripple} ease-out infinite`,
      'echo-glow-breath': `echo-glow-breath ${duration.ripple} ease-in-out infinite`,
      'echo-press': 'echo-press 450ms ease-out forwards',
      // As long as the real press and hold (HOLD_MS).
      'echo-hold-fill': 'echo-hold-fill 1s linear forwards',
      'echo-flash': 'echo-flash 900ms ease-out',
      'echo-demo-stage': `echo-demo-stage ${duration.demo} ease-in-out infinite`,
      'echo-demo-select': `echo-demo-select ${duration.demo} ease-out infinite`,
      'echo-demo-pointer': `echo-demo-pointer ${duration.demo} ease-in-out infinite`,
      'echo-demo-ring': `echo-demo-ring ${duration.demo} linear infinite`,
      'echo-demo-menu': `echo-demo-menu ${duration.demo} ease-out infinite`,
      'echo-demo-pick': `echo-demo-pick ${duration.demo} ease-out infinite`,
      'echo-demo-key-1': `echo-demo-key-1 ${duration.demo} ease-out infinite`,
      'echo-demo-key-2': `echo-demo-key-2 ${duration.demo} ease-out infinite`,
      'echo-demo-key-3': `echo-demo-key-3 ${duration.demo} ease-out infinite`,
      'echo-demo-chip': `echo-demo-chip ${duration.demo} ease-out infinite`,
      'echo-shimmer': `echo-shimmer ${duration.shimmer} linear infinite`,
    },
  },
};

module.exports = { theme, themeVariables };
