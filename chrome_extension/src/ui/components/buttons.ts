/** Button looks shared by the extension pages, so every action reads the same. */
const BASE =
  '_echo_$_shrink-0 _echo_$_whitespace-nowrap _echo_$_rounded-lg _echo_$_px-3 _echo_$_py-1.5 _echo_$_text-xs _echo_$_font-semibold _echo_$_transition disabled:_echo_$_cursor-default disabled:_echo_$_opacity-60 focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-focus';

export const PRIMARY_BUTTON = `${BASE} _echo_$_bg-accent _echo_$_text-on-accent _echo_$_shadow-sm hover:_echo_$_bg-accent-hover`;

export const SECONDARY_BUTTON = `${BASE} _echo_$_border _echo_$_border-line-strong _echo_$_bg-surface _echo_$_text-fg-secondary hover:_echo_$_bg-hover`;

export const DANGER_BUTTON = `${BASE} _echo_$_bg-danger _echo_$_text-on-danger _echo_$_shadow-sm hover:_echo_$_bg-danger-hover`;
