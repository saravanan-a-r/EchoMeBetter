import type { KeyPlatform } from '../../shared/shortcuts';
import { STYLES } from '../../shared/styles';
import { StyleGlyph } from '../brand/StyleGlyph';
import { ShortcutKeys } from './ShortcutKeys';

/**
 * `dense`: one bordered card with a row per style, sized for the 600px-high toolbar popup.
 * `shortcuts`: show each style's shortcut, written for that platform (only while shortcuts work).
 */
export function StyleList({ columns = 1, dense = false, shortcuts }: { columns?: 1 | 2 | 3; dense?: boolean; shortcuts?: KeyPlatform }) {
  if (dense) {
    return (
      <ul className="_echo_$_divide-y _echo_$_divide-ink-100 _echo_$_overflow-hidden _echo_$_rounded-xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white dark:_echo_$_divide-ink-800 dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900">
        {STYLES.map((style) => (
          <li key={style.id} className="_echo_$_flex _echo_$_items-center _echo_$_gap-2.5 _echo_$_px-3 _echo_$_py-2">
            <span className="_echo_$_flex _echo_$_h-6 _echo_$_w-6 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-md _echo_$_bg-echo-50 _echo_$_text-echo-600 dark:_echo_$_bg-echo-950 dark:_echo_$_text-echo-300">
              <StyleGlyph style={style.id} size={14} />
            </span>
            <span className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">{style.label}</span>
            {/* A row has room for the hint or the shortcut; the shortcut is the one to learn here. */}
            {shortcuts ? (
              <span title={style.hint} className="_echo_$_ml-auto">
                <ShortcutKeys style={style.id} platform={shortcuts} compact />
              </span>
            ) : (
              <span className="_echo_$_ml-auto _echo_$_truncate _echo_$_text-xs _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{style.hint}</span>
            )}
          </li>
        ))}
      </ul>
    );
  }
  const grid = columns === 1 ? '' : columns === 2 ? 'sm:_echo_$_grid-cols-2' : 'sm:_echo_$_grid-cols-2 lg:_echo_$_grid-cols-3';
  return (
    <ul className={`_echo_$_grid _echo_$_gap-2 ${grid}`}>
      {STYLES.map((style) => (
        <li
          key={style.id}
          className="_echo_$_flex _echo_$_items-center _echo_$_gap-3 _echo_$_rounded-xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_px-3 _echo_$_py-2.5 dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900"
        >
          <span className="_echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg _echo_$_bg-echo-50 _echo_$_text-echo-600 dark:_echo_$_bg-echo-950 dark:_echo_$_text-echo-300">
            <StyleGlyph style={style.id} />
          </span>
          <span className="_echo_$_min-w-0 _echo_$_flex-1">
            <span className="_echo_$_block _echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">{style.label}</span>
            <span className="_echo_$_block _echo_$_text-xs _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{style.hint}</span>
          </span>
          {shortcuts ? <ShortcutKeys style={style.id} platform={shortcuts} /> : null}
        </li>
      ))}
    </ul>
  );
}
