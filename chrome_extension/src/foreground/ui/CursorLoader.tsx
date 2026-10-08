import { useLayoutEffect, useRef } from 'react';
import { styleLabel } from '../../shared/styles';
import { EchoLoader } from '../../ui/brand/EchoLoader';
import type { WorkingState } from './overlayStore';
import { followPointer } from './pointerFollower';

export function workingLabel(working: WorkingState): string {
  switch (working.phase) {
    case 'loading-model': {
      const percent = Math.round((working.progress ?? 0) * 100);
      return percent > 0 ? `Waking up the writing model · ${percent}%` : 'Waking up the writing model…';
    }
    case 'rewriting':
      return `Rewriting · ${styleLabel(working.style)}`;
    case 'starting':
      return 'Getting ready…';
  }
}

/** The pill that rides along with the mouse pointer while a rewrite runs. */
export function CursorLoader({ working }: { working: WorkingState }) {
  const ref = useRef<HTMLDivElement>(null);
  const { x, y } = working.origin;

  useLayoutEffect(() => {
    if (!ref.current) return undefined;
    return followPointer(ref.current, { x, y }, ref.current.ownerDocument.defaultView ?? window);
  }, [x, y]);

  const label = workingLabel(working);
  return (
    <div
      ref={ref}
      role="status"
      aria-live="polite"
      aria-label={label}
      className="_echo_$_fixed _echo_$_left-0 _echo_$_top-0 _echo_$_flex _echo_$_items-center _echo_$_gap-2.5 _echo_$_rounded-full _echo_$_bg-surface _echo_$_py-2 _echo_$_pl-2 _echo_$_pr-2.5 _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15 _echo_$_will-change-transform"
    >
      <span className="_echo_$_flex _echo_$_h-7 _echo_$_w-7 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-full _echo_$_bg-sheen/10">
        <EchoLoader size={20} label={label} />
      </span>
      <span className="_echo_$_whitespace-nowrap">{label}</span>
      <kbd className="_echo_$_ml-0.5 _echo_$_rounded-md _echo_$_border _echo_$_border-sheen/25 _echo_$_bg-sheen/10 _echo_$_px-1.5 _echo_$_py-0.5 _echo_$_font-sans _echo_$_text-[12px] _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-fg-secondary">
        Esc
      </kbd>
    </div>
  );
}
