import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { styleLabel } from '../../shared/styles';
import { Logo } from '../../ui/brand/Logo';
import type { AnchorRect, ToastState, WorkingState } from './overlayStore';

export const CARD_WIDTH = 360;
const GAP = 10;

/** Below the edited text when there is room, otherwise above it; centred at the bottom without an anchor. */
export function toastPosition(anchor: AnchorRect | null, viewport: { width: number; height: number }): CSSProperties {
  if (!anchor) return { left: Math.max(GAP, (viewport.width - CARD_WIDTH) / 2), bottom: 24 };
  const left = Math.min(Math.max(GAP, anchor.left), Math.max(GAP, viewport.width - CARD_WIDTH - GAP));
  if (anchor.bottom + GAP + 96 <= viewport.height) return { left, top: Math.max(GAP, anchor.bottom + GAP) };
  return { left, bottom: Math.max(GAP, viewport.height - anchor.top + GAP) };
}

/** The most of the window the preview may take across, as it grows to fit the rewrite. */
export const PREVIEW_MAX_WIDTH_SHARE = 0.7;
/** The preview's text grows to this tall, then scrolls. */
export const PREVIEW_MAX_TEXT_HEIGHT = 120;

/** Widths the preview may take: a toast's width at the least, 70% of the window at the most (never less than the least). */
export function previewWidthRange(viewport: { width: number }): { min: number; max: number } {
  const min = Math.min(CARD_WIDTH, viewport.width - 2 * GAP);
  return { min, max: Math.max(min, Math.floor(viewport.width * PREVIEW_MAX_WIDTH_SHARE)) };
}

/**
 * Where the preview sits and how wide it is. It is as wide as its text
 * wants, between the range's two ends, so it grows to the right as the
 * rewrite does; where that would run off the window it moves left by
 * exactly the overflow. That shift is a `translate` (a percentage in it
 * means the card's own width, which only the browser knows), so no JavaScript
 * measures the card as it grows. It is `translate`, not `transform`, because
 * the card's pop-in animation holds `transform`.
 */
export function previewStyle(anchor: AnchorRect | null, viewport: { width: number; height: number }): CSSProperties {
  const { min, max } = previewWidthRange(viewport);
  const place = toastPosition(anchor, viewport);
  const left = place.left as number;
  return { ...place, width: 'max-content', minWidth: min, maxWidth: max, translate: `min(0px, calc(${viewport.width - GAP - left}px - 100%))` };
}

const TONE_ACCENT: Record<ToastState['tone'], string> = {
  success: '_echo_$_bg-success',
  info: '_echo_$_bg-info',
  error: '_echo_$_bg-danger',
};

/** A working rewrite whose text has started to arrive. */
export type Streaming = WorkingState & { readonly preview: string };

const BUTTON =
  '_echo_$_cursor-pointer _echo_$_rounded-lg _echo_$_border-0 _echo_$_bg-accent _echo_$_px-3.5 _echo_$_py-1.5 _echo_$_font-sans _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-on-accent hover:_echo_$_bg-accent-hover focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-focus';

/**
 * The one card the overlay shows beside the edited text: the rewrite as the
 * model writes it, then, in place, the toast with the outcome (how long it
 * took, Undo, how long waking the model took). A toast that comes without a
 * preview (live preview off, or nothing written yet) uses the same card.
 *
 * One component in one place on purpose: React keeps the element as the
 * preview turns into the toast, so the card changes its content where it is
 * instead of closing and another opening.
 *
 * While previewing it takes no clicks and screen readers hear its title, not
 * every word; the newest text stays in view by layout alone (bottom-aligned,
 * older lines clipped at the top). It is redrawn about ten times a second
 * while the model runs, often on the same GPU: on a layer of its own with its
 * layout and paint contained, an update repaints the card alone.
 */
export function OverlayCard({ toast, streaming, onDismiss }: { toast: ToastState | null; streaming: Streaming | null; onDismiss: () => void }) {
  const [hovered, setHovered] = useState(false);
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;

  useEffect(() => {
    if (!toast || toast.durationMs === 0 || hovered) return undefined;
    const timer = setTimeout(() => dismiss.current(), toast.durationMs);
    return () => clearTimeout(timer);
  }, [toast?.id, toast?.durationMs, hovered]);

  const viewport = { width: window.innerWidth, height: window.innerHeight };
  const anchor = toast ? toast.anchor : (streaming?.anchor ?? null);
  const title = toast ? toast.title : `Rewriting · ${styleLabel(streaming!.style)}`;

  return (
    <div
      role={toast?.tone === 'error' ? 'alert' : 'status'}
      style={{
        ...(toast ? { ...toastPosition(anchor, viewport), width: Math.min(CARD_WIDTH, viewport.width - 2 * GAP) } : previewStyle(anchor, viewport)),
        contain: 'layout paint',
      }}
      // Keep focus (and the caret) in the user's editor when a toast button is pressed.
      onMouseDown={(event) => event.preventDefault()}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`_echo_$_pointer-events-auto _echo_$_fixed _echo_$_flex _echo_$_animate-echo-pop-in _echo_$_overflow-hidden _echo_$_rounded-2xl _echo_$_bg-surface _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15 _echo_$_will-change-transform`}
    >
      <span aria-hidden="true" className={`_echo_$_w-1.5 _echo_$_shrink-0 ${toast ? TONE_ACCENT[toast.tone] : '_echo_$_bg-accent'}`} />
      <div className="_echo_$_flex _echo_$_min-w-0 _echo_$_flex-1 _echo_$_items-start _echo_$_gap-3 _echo_$_p-4">
        <span className="_echo_$_mt-0.5 _echo_$_shrink-0">
          <Logo size={24} />
        </span>
        <div className="_echo_$_min-w-0 _echo_$_flex-1">
          <p className="_echo_$_m-0 _echo_$_text-[15px] _echo_$_font-semibold _echo_$_leading-[22px] _echo_$_text-fg">{title}</p>
          {toast ? <ToastBody toast={toast} onDismiss={onDismiss} /> : <PreviewText text={streaming!.preview} />}
        </div>
        {toast ? (
          <button
            type="button"
            aria-label="Dismiss"
            onClick={onDismiss}
            className="_echo_$_m-[-4px] _echo_$_cursor-pointer _echo_$_rounded-md _echo_$_border-0 _echo_$_bg-transparent _echo_$_p-1.5 _echo_$_text-fg-soft hover:_echo_$_bg-sheen/10 hover:_echo_$_text-fg"
          >
            <svg width="16" height="16" viewBox="0 0 14 14" aria-hidden="true">
              <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        ) : (
          <kbd className="_echo_$_mt-0.5 _echo_$_shrink-0 _echo_$_rounded-md _echo_$_border _echo_$_border-solid _echo_$_border-sheen/25 _echo_$_bg-sheen/10 _echo_$_px-1.5 _echo_$_py-0.5 _echo_$_font-sans _echo_$_text-[12px] _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-fg-secondary">
            Esc
          </kbd>
        )}
      </div>
    </div>
  );
}

/** Within this many pixels of the end counts as being at the end (scroll positions are fractional on some screens). */
const AT_END_SLACK = 4;

/**
 * The rewrite so far, with a caret where the model is writing. It grows to
 * PREVIEW_MAX_TEXT_HEIGHT, then scrolls. It follows the newest text until the
 * reader scrolls up to read from the start, and follows again once they
 * scroll back to the end.
 */
function PreviewText({ text }: { text: string }) {
  const scroller = useRef<HTMLDivElement>(null);
  const following = useRef(true);

  useLayoutEffect(() => {
    const element = scroller.current;
    if (element && following.current) element.scrollTop = element.scrollHeight;
  }, [text]);

  return (
    <div
      ref={scroller}
      aria-hidden="true"
      onScroll={(event) => {
        const { scrollHeight, scrollTop, clientHeight } = event.currentTarget;
        following.current = scrollHeight - scrollTop - clientHeight <= AT_END_SLACK;
      }}
      style={{ maxHeight: PREVIEW_MAX_TEXT_HEIGHT }}
      className="echo-scroll _echo_$_mt-1 _echo_$_overflow-y-auto _echo_$_overscroll-contain"
    >
      <p className="_echo_$_m-0 _echo_$_whitespace-pre-wrap _echo_$_break-words _echo_$_text-[14px] _echo_$_leading-5 _echo_$_text-fg-secondary">
        {text}
        <span className="_echo_$_ml-0.5 _echo_$_inline-block _echo_$_h-[1em] _echo_$_w-0.5 _echo_$_translate-y-[0.15em] _echo_$_rounded-full _echo_$_bg-accent _echo_$_animate-echo-wave" />
      </p>
    </div>
  );
}

function ToastBody({ toast, onDismiss }: { toast: ToastState; onDismiss: () => void }) {
  return (
    <>
      {toast.message ? <p className="_echo_$_m-0 _echo_$_mt-1 _echo_$_text-[14px] _echo_$_leading-5 _echo_$_text-fg-secondary">{toast.message}</p> : null}
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
              className={BUTTON}
            >
              {action.label}
            </button>
          ))}
        </div>
      ) : null}
      {toast.note ? (
        <div className="_echo_$_mt-3 _echo_$_border-0 _echo_$_border-t _echo_$_border-solid _echo_$_border-sheen/15 _echo_$_pt-3">
          <p className="_echo_$_m-0 _echo_$_text-[13px] _echo_$_leading-5 _echo_$_text-fg-soft">{toast.note.text}</p>
          <button
            type="button"
            onClick={() => {
              toast.note?.action.run();
              onDismiss();
            }}
            className="_echo_$_mt-2 _echo_$_cursor-pointer _echo_$_rounded-lg _echo_$_border _echo_$_border-solid _echo_$_border-sheen/25 _echo_$_bg-sheen/10 _echo_$_px-3 _echo_$_py-1.5 _echo_$_font-sans _echo_$_text-[13px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-fg hover:_echo_$_bg-sheen/20 focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-focus"
          >
            {toast.note.action.label}
          </button>
        </div>
      ) : null}
    </>
  );
}
