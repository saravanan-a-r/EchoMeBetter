import { useEffect, useRef, useState, type CSSProperties } from 'react';
import { Logo } from '../../ui/brand/Logo';
import type { AnchorRect, ToastState } from './overlayStore';

const TOAST_WIDTH = 360;
const GAP = 10;

/** Below the edited text when there is room, otherwise above it; centred at the bottom without an anchor. */
export function toastPosition(anchor: AnchorRect | null, viewport: { width: number; height: number }): CSSProperties {
  if (!anchor) return { left: Math.max(GAP, (viewport.width - TOAST_WIDTH) / 2), bottom: 24 };
  const left = Math.min(Math.max(GAP, anchor.left), Math.max(GAP, viewport.width - TOAST_WIDTH - GAP));
  if (anchor.bottom + GAP + 96 <= viewport.height) return { left, top: Math.max(GAP, anchor.bottom + GAP) };
  return { left, bottom: Math.max(GAP, viewport.height - anchor.top + GAP) };
}

const TONE_ACCENT: Record<ToastState['tone'], string> = {
  success: '_echo_$_bg-better-400',
  info: '_echo_$_bg-echo-400',
  error: '_echo_$_bg-danger-400',
};

export function Toast({ toast, onDismiss }: { toast: ToastState; onDismiss: () => void }) {
  const [hovered, setHovered] = useState(false);
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;

  useEffect(() => {
    if (toast.durationMs === 0 || hovered) return undefined;
    const timer = setTimeout(() => dismiss.current(), toast.durationMs);
    return () => clearTimeout(timer);
  }, [toast.id, toast.durationMs, hovered]);

  const position = toastPosition(toast.anchor, { width: window.innerWidth, height: window.innerHeight });

  return (
    <div
      role={toast.tone === 'error' ? 'alert' : 'status'}
      style={{ ...position, width: Math.min(TOAST_WIDTH, window.innerWidth - 2 * GAP) }}
      // Keep focus (and the caret) in the user's editor when a toast button is pressed.
      onMouseDown={(event) => event.preventDefault()}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="_echo_$_pointer-events-auto _echo_$_fixed _echo_$_flex _echo_$_animate-echo-pop-in _echo_$_overflow-hidden _echo_$_rounded-2xl _echo_$_bg-ink-900 _echo_$_text-white _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-white/15"
    >
      <span aria-hidden="true" className={`_echo_$_w-1.5 _echo_$_shrink-0 ${TONE_ACCENT[toast.tone]}`} />
      <div className="_echo_$_flex _echo_$_min-w-0 _echo_$_flex-1 _echo_$_items-start _echo_$_gap-3 _echo_$_p-4">
        <span className="_echo_$_mt-0.5 _echo_$_shrink-0">
          <Logo size={24} />
        </span>
        <div className="_echo_$_min-w-0 _echo_$_flex-1">
          <p className="_echo_$_m-0 _echo_$_text-[15px] _echo_$_font-semibold _echo_$_leading-[22px] _echo_$_text-white">{toast.title}</p>
          {toast.message ? (
            <p className="_echo_$_m-0 _echo_$_mt-1 _echo_$_text-[14px] _echo_$_leading-5 _echo_$_text-ink-100">{toast.message}</p>
          ) : null}
          {toast.actions.length > 0 ? (
            <div className="_echo_$_mt-3 _echo_$_flex _echo_$_gap-2">
              {toast.actions.map((action) => (
                <button
                  key={action.label}
                  type="button"
                  onClick={() => {
                    action.run();
                    onDismiss();
                  }}
                  className="_echo_$_cursor-pointer _echo_$_rounded-lg _echo_$_border-0 _echo_$_bg-better-300 _echo_$_px-3.5 _echo_$_py-1.5 _echo_$_font-sans _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-ink-950 hover:_echo_$_bg-better-200 focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-better-200"
                >
                  {action.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <button
          type="button"
          aria-label="Dismiss"
          onClick={onDismiss}
          className="_echo_$_m-[-4px] _echo_$_cursor-pointer _echo_$_rounded-md _echo_$_border-0 _echo_$_bg-transparent _echo_$_p-1.5 _echo_$_text-ink-200 hover:_echo_$_bg-white/10 hover:_echo_$_text-white"
        >
          <svg width="16" height="16" viewBox="0 0 14 14" aria-hidden="true">
            <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}
