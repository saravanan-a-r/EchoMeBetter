import { shortcutKeys, shortcutLabel, spokenShortcut, type KeyPlatform } from '../../shared/shortcuts';
import type { StyleId } from '../../shared/styles';

const KEY =
  '_echo_$_inline-flex _echo_$_min-w-[22px] _echo_$_items-center _echo_$_justify-center _echo_$_rounded-md _echo_$_border _echo_$_border-b-2 _echo_$_border-ink-200 _echo_$_bg-white _echo_$_px-1.5 _echo_$_py-px _echo_$_font-sans _echo_$_text-xs _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-ink-800 dark:_echo_$_border-ink-600 dark:_echo_$_bg-ink-800 dark:_echo_$_text-ink-100';

/**
 * A style's shortcut, written the way the user's platform writes it: one key
 * cap per key, or `compact` as a single cap for tight rows.
 */
export function ShortcutKeys({ style, platform, compact = false }: { style: StyleId; platform: KeyPlatform; compact?: boolean }) {
  if (compact) {
    return (
      <kbd aria-label={spokenShortcut(style, platform)} className={`${KEY} _echo_$_shrink-0 _echo_$_whitespace-nowrap _echo_$_tracking-wide`}>
        {shortcutLabel(style, platform)}
      </kbd>
    );
  }
  return (
    <span role="img" aria-label={spokenShortcut(style, platform)} className="_echo_$_inline-flex _echo_$_shrink-0 _echo_$_items-center _echo_$_gap-1">
      {shortcutKeys(style, platform).map((key) => (
        <kbd key={key} aria-hidden="true" className={KEY}>
          {key}
        </kbd>
      ))}
    </span>
  );
}
