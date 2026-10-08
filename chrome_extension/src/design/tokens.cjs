/**
 * The design tokens in tokens.json, resolved: every "{path.to.token}"
 * reference replaced by its value, and each theme complete (a theme that
 * extends another gets the other's tokens, then its own).
 *
 * Plain CommonJS with no dependencies, so the Tailwind configs (Node) and
 * the extension's own code (bundled) read the same values.
 *
 * Restyling the UI means editing tokens.json, not the components:
 * components use theme tokens through Tailwind (bg-surface, text-fg-muted,
 * border-line, bg-accent, ...), which follow the light, dark and inverse
 * themes on their own, so they carry no dark: colour variants. Palette
 * classes (text-echo-500) are for brand artwork only. Code outside CSS
 * reads values from here, e.g. `brand.badge` for the toolbar badge.
 */
const source = require('./tokens.json');

const REFERENCE = /\{([^{}]+)\}/g;

function lookup(path) {
  let node = source;
  for (const key of path.split('.')) {
    node = node?.[key];
    if (node === undefined) throw new Error(`design tokens: "{${path}}" names no token`);
  }
  if (typeof node !== 'object' || !('$value' in node)) throw new Error(`design tokens: "{${path}}" is a group, not a token`);
  return node.$value;
}

/** A token's value with its references resolved (references may nest, and may sit inside a longer string). */
function resolve(value, seen = []) {
  if (Array.isArray(value)) return value.map((item) => resolve(item, seen));
  if (typeof value !== 'string') return value;
  return value.replace(REFERENCE, (_match, path) => {
    if (seen.includes(path)) throw new Error(`design tokens: circular reference through "{${path}}"`);
    return resolve(lookup(path), [...seen, path]);
  });
}

/** The tokens of a group, by name, resolved; `$`-keys are metadata. */
function values(group) {
  return Object.fromEntries(
    Object.entries(group)
      .filter(([key]) => !key.startsWith('$'))
      .map(([key, token]) => [key, resolve(token.$value)]),
  );
}

function theme(name, chain = []) {
  if (chain.includes(name)) throw new Error(`design tokens: theme "${name}" extends itself`);
  const group = source.theme[name];
  if (!group) throw new Error(`design tokens: no theme "${name}"`);
  const base = group.$extensions?.['com.echomebetter.extends'];
  return { ...(base ? theme(base, [...chain, name]) : {}), ...values(group) };
}

/** #RGB, #RRGGBB or #RRGGBBAA as CSS rgb() channels ("118 80 255" or "118 80 255 / 0.4"). */
function channels(hex) {
  const match = /^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i.exec(hex);
  if (!match) throw new Error(`design tokens: "${hex}" is not a hex colour`);
  let digits = match[1];
  if (digits.length === 3) digits = [...digits].map((digit) => digit + digit).join('');
  const [r, g, b] = [0, 2, 4].map((offset) => parseInt(digits.slice(offset, offset + 2), 16));
  if (digits.length === 6) return `${r} ${g} ${b}`;
  const alpha = Math.round((parseInt(digits.slice(6, 8), 16) / 255) * 100) / 100;
  return `${r} ${g} ${b} / ${alpha}`;
}

function palette() {
  const groups = Object.entries(source.palette).filter(([key]) => !key.startsWith('$'));
  return Object.fromEntries(groups.map(([key, group]) => [key, '$value' in group ? group.$value : values(group)]));
}

const themes = Object.fromEntries(
  Object.keys(source.theme)
    .filter((key) => !key.startsWith('$'))
    .map((name) => [name, theme(name)]),
);

module.exports = {
  palette: palette(),
  themes,
  gradient: values(source.gradient),
  brand: values(source.brand),
  font: values(source.font),
  shadow: values(source.shadow),
  motion: { duration: values(source.motion.duration), easing: values(source.motion.easing) },
  channels,
};
