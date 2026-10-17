import type { KeyPlatform } from '../../shared/shortcuts';
import { STYLES, type StyleDescriptor } from '../../shared/styles';
import { StyleGlyph } from '../brand/StyleGlyph';
import { ShortcutKeys } from './ShortcutKeys';

export interface StyleListProps {
  readonly columns?: 1 | 2 | 3;
  /** Show each style's shortcut, written for that platform (only while shortcuts work). */
  readonly shortcuts?: KeyPlatform;
}

function Label({ style, className }: { style: StyleDescriptor; className: string }) {
  return <span className={className}>{style.label}</span>;
}

export function StyleList(props: StyleListProps) {
  const { columns = 1, shortcuts } = props;
  const grid = columns === 1 ? '' : columns === 2 ? 'sm:_echo_$_grid-cols-2' : 'sm:_echo_$_grid-cols-2 lg:_echo_$_grid-cols-3';
  return (
    <ul className={`_echo_$_grid _echo_$_gap-2 ${grid}`}>
      {STYLES.map((style) => (
        <li
          key={style.id}
          className="_echo_$_flex _echo_$_items-center _echo_$_gap-3 _echo_$_rounded-xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_px-3 _echo_$_py-2.5"
        >
          <span className="_echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg _echo_$_bg-accent-subtle _echo_$_text-accent-fg">
            <StyleGlyph style={style.id} />
          </span>
          <span className="_echo_$_min-w-0 _echo_$_flex-1">
            <Label style={style} className="_echo_$_block _echo_$_text-sm _echo_$_font-semibold _echo_$_text-fg" />
            <span className="_echo_$_block _echo_$_text-xs _echo_$_text-fg-muted">{style.hint}</span>
          </span>
          {shortcuts ? <ShortcutKeys style={style.id} platform={shortcuts} /> : null}
        </li>
      ))}
    </ul>
  );
}
