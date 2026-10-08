import { useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { STYLES } from '../../shared/styles';
import { Logo } from '../../ui/brand/Logo';
import { StyleGlyph } from '../../ui/brand/StyleGlyph';
import type { Point, StyleMenuState } from './overlayStore';

const GAP = 8;
/** Down and to the right of the pointer, clear of it, so letting go of the hold lands on nothing. */
const OFFSET = { x: 10, y: 14 } as const;

/** Beside the point, flipped to its other side where the viewport ends, and never off screen. */
export function menuPosition(point: Point, size: { width: number; height: number }, viewport: { width: number; height: number }): CSSProperties {
  const right = point.x + OFFSET.x;
  const below = point.y + OFFSET.y;
  const left = right + size.width + GAP <= viewport.width ? right : point.x - OFFSET.x - size.width;
  const top = below + size.height + GAP <= viewport.height ? below : point.y - OFFSET.y - size.height;
  return {
    left: Math.max(GAP, Math.min(left, viewport.width - size.width - GAP)),
    top: Math.max(GAP, Math.min(top, viewport.height - size.height - GAP)),
  };
}

/** The styles, beside where the text was held: pick with the mouse, or ↑ ↓ and Enter, or a number key. */
export function StyleMenu({ menu }: { menu: StyleMenuState }) {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<CSSProperties>({ left: menu.point.x + OFFSET.x, top: menu.point.y + OFFSET.y, visibility: 'hidden' });

  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const view = element.ownerDocument.defaultView ?? window;
    const { width, height } = element.getBoundingClientRect();
    setPosition(menuPosition(menu.point, { width, height }, { width: view.innerWidth, height: view.innerHeight }));
  }, [menu.point]);

  const items = menu.styles.map((id) => STYLES.find((style) => style.id === id)!);
  return (
    <div
      ref={ref}
      role="menu"
      aria-label="Rewrite as"
      style={position}
      // Keep focus, and with it the selection, in the user's editor.
      onMouseDown={(event) => event.preventDefault()}
      className="_echo_$_pointer-events-auto _echo_$_fixed _echo_$_w-[272px] _echo_$_animate-echo-pop-in _echo_$_rounded-2xl _echo_$_bg-surface _echo_$_p-1.5 _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15"
    >
      <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-2 _echo_$_px-2 _echo_$_pb-1.5 _echo_$_pt-1">
        <Logo size={16} />
        <span className="_echo_$_text-[12px] _echo_$_font-semibold _echo_$_uppercase _echo_$_leading-4 _echo_$_tracking-wider _echo_$_text-fg-muted">Rewrite as</span>
        <kbd className="_echo_$_ml-auto _echo_$_rounded-md _echo_$_border _echo_$_border-solid _echo_$_border-sheen/25 _echo_$_bg-sheen/10 _echo_$_px-1.5 _echo_$_py-0.5 _echo_$_font-sans _echo_$_text-[11px] _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-fg-secondary">
          Esc
        </kbd>
      </div>
      {items.map((style, index) => {
        const active = index === menu.active;
        return (
          <button
            key={style.id}
            type="button"
            role="menuitem"
            data-active={active || undefined}
            onClick={() => menu.pick(style.id)}
            onMouseEnter={() => menu.setActive(index)}
            className={`_echo_$_flex _echo_$_w-full _echo_$_cursor-pointer _echo_$_items-center _echo_$_gap-2.5 _echo_$_rounded-xl _echo_$_border-0 _echo_$_px-2 _echo_$_py-1.5 _echo_$_text-left _echo_$_font-sans _echo_$_text-fg ${
              active ? '_echo_$_bg-sheen/10' : '_echo_$_bg-transparent'
            }`}
          >
            <span
              className={`_echo_$_flex _echo_$_h-7 _echo_$_w-7 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg ${
                active ? '_echo_$_bg-accent _echo_$_text-on-accent' : '_echo_$_bg-sheen/10 _echo_$_text-accent'
              }`}
            >
              <StyleGlyph style={style.id} size={16} />
            </span>
            <span className="_echo_$_min-w-0 _echo_$_flex-1">
              <span className="_echo_$_block _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5">{style.label}</span>
              <span className="_echo_$_block _echo_$_truncate _echo_$_text-[12px] _echo_$_leading-4 _echo_$_text-fg-muted">{style.hint}</span>
            </span>
            <kbd aria-hidden="true" className="_echo_$_shrink-0 _echo_$_font-sans _echo_$_text-[12px] _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-fg-subtle">
              {index + 1}
            </kbd>
          </button>
        );
      })}
    </div>
  );
}
