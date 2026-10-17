import type { CSSProperties } from 'react';
import type { AnchorRect } from './overlayStore';

/** The space kept between a card and the edge of the window, and between it and the text it belongs to. */
export const GAP = 10;

export const BUTTON =
  '_echo_$_cursor-pointer _echo_$_rounded-lg _echo_$_border-0 _echo_$_bg-accent _echo_$_px-3.5 _echo_$_py-1.5 _echo_$_font-sans _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-on-accent hover:_echo_$_bg-accent-hover disabled:_echo_$_cursor-default disabled:_echo_$_opacity-70 focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-focus';

export const SECONDARY_BUTTON =
  '_echo_$_cursor-pointer _echo_$_rounded-lg _echo_$_border _echo_$_border-solid _echo_$_border-sheen/25 _echo_$_bg-sheen/10 _echo_$_px-3 _echo_$_py-1.5 _echo_$_font-sans _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-fg hover:_echo_$_bg-sheen/20 focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-focus';

/** Where a tall card sits: below the edited text if it fits, else above it, else against the window's bottom edge. */
export function offerPosition(anchor: AnchorRect | null, viewport: { width: number; height: number }, width: number, height: number): CSSProperties {
  if (!anchor) return { left: Math.max(GAP, (viewport.width - width) / 2), bottom: 24 };
  const left = Math.min(Math.max(GAP, anchor.left), Math.max(GAP, viewport.width - width - GAP));
  if (anchor.bottom + GAP + height <= viewport.height) return { left, top: Math.max(GAP, anchor.bottom + GAP) };
  if (anchor.top - GAP - height >= 0) return { left, bottom: Math.max(GAP, viewport.height - anchor.top + GAP) };
  return { left, bottom: GAP };
}
