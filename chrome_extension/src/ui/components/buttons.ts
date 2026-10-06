/** Button looks shared by the extension pages, so every action reads the same. */
const BASE =
  '_echo_$_shrink-0 _echo_$_whitespace-nowrap _echo_$_rounded-lg _echo_$_px-3 _echo_$_py-1.5 _echo_$_text-xs _echo_$_font-semibold _echo_$_transition disabled:_echo_$_cursor-default disabled:_echo_$_opacity-60 focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-echo-500';

export const PRIMARY_BUTTON = `${BASE} _echo_$_bg-echo-600 _echo_$_text-white _echo_$_shadow-sm hover:_echo_$_bg-echo-700`;

export const SECONDARY_BUTTON = `${BASE} _echo_$_border _echo_$_border-ink-200 _echo_$_bg-white _echo_$_text-ink-700 hover:_echo_$_bg-ink-50 dark:_echo_$_border-ink-700 dark:_echo_$_bg-ink-900 dark:_echo_$_text-ink-100 dark:hover:_echo_$_bg-ink-800`;

export const DANGER_BUTTON = `${BASE} _echo_$_bg-danger-500 _echo_$_text-white _echo_$_shadow-sm hover:_echo_$_bg-danger-600`;
