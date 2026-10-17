import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { shortcutKeys, shortcutLabel } from '../../shared/shortcuts';
import { styleLabel } from '../../shared/styles';
import { Logo } from '../../ui/brand/Logo';
import { BUTTON, GAP, offerPosition, SECONDARY_BUTTON } from './cardKit';
import type { SiteAccessOffer, ToastState } from './overlayStore';

export const OFFER_WIDTH = 460;
/** Below this the two demos stack instead of sitting side by side. */
const TWO_COLUMNS_FROM = 420;
/** The card's height, close enough to decide whether it fits below the text or above it. */
const OFFER_HEIGHT = 450;
const STACKED_EXTRA_HEIGHT = 190;

const STAGE = '_echo_$_relative _echo_$_h-[120px] _echo_$_overflow-hidden _echo_$_rounded-lg _echo_$_bg-canvas _echo_$_ring-1 _echo_$_ring-sheen/15';

/** What the card says for the features it is about (both, or the one the user kept on). */
function copy(offer: SiteAccessOffer) {
  const both = offer.hold && offer.shortcuts;
  return {
    title: both ? 'Rewrite faster, without the right-click menu' : offer.hold ? 'Rewrite with a press and hold' : 'Rewrite with a keyboard shortcut',
    description: `One-time permission to turn on ${both ? 'these two features' : offer.hold ? 'press and hold' : 'keyboard shortcuts'}. Chrome will ask you to confirm.`,
    onTitle: both ? 'Hold and shortcuts are on' : offer.hold ? 'Press and hold is on' : 'Keyboard shortcuts are on',
    onDescription: 'They work in any text box now. Select some text and try one.',
    offTitle: 'Not turned on',
    offDescription: "Chrome didn't get your permission. You can allow websites any time from the EchoMeBetter popup, under Settings.",
  };
}

type Phase = 'offer' | 'asking' | 'on' | 'declined';

/**
 * "Text replaced" grown into an invitation to allow press and hold and the
 * keyboard shortcuts: the toast's own line (title, Undo, close) stays on
 * top, and below it two looping demos show what Allow turns on, the way the
 * welcome page does. One Chrome permission serves both, so there is one button.
 *
 * Right-click needs no permission, which is why this waits until the user has
 * seen the product work (see shared/siteAccessOffer.ts for when).
 */
export function AccessOfferCard({
  toast,
  offer,
  viewport,
  header,
  onDismiss,
  onHover,
  onAsking,
}: {
  toast: ToastState;
  offer: SiteAccessOffer;
  viewport: { width: number; height: number };
  /** The toast's first line, so Undo stays where the user just saw it. */
  header: ReactNode;
  onDismiss: () => void;
  onHover: (hovered: boolean) => void;
  /** Chrome's prompt is up; the card must stay until it is answered. */
  onAsking: (asking: boolean) => void;
}) {
  const [phase, setPhase] = useState<Phase>('offer');
  const titleId = useId();
  const mounted = useRef(true);
  const dismiss = useRef(onDismiss);
  dismiss.current = onDismiss;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    onAsking(phase === 'asking');
    return () => onAsking(false);
  }, [phase, onAsking]);
  useEffect(() => {
    if (phase !== 'on' && phase !== 'declined') return undefined;
    const timer = setTimeout(() => dismiss.current(), phase === 'on' ? 5000 : 9000);
    return () => clearTimeout(timer);
  }, [phase]);

  const allow = () => {
    setPhase('asking');
    offer.allow().then(
      (result) => {
        if (!mounted.current) return;
        if (result === 'opened') dismiss.current();
        else setPhase(result === 'granted' ? 'on' : 'declined');
      },
      () => mounted.current && setPhase('declined'),
    );
  };

  const say = copy(offer);
  const demos = [offer.hold ? 'hold' : null, offer.shortcuts ? 'shortcuts' : null].filter((demo) => demo !== null);
  const width = Math.min(OFFER_WIDTH, viewport.width - 2 * GAP);
  const columns = demos.length === 2 && width >= TWO_COLUMNS_FROM ? 2 : 1;
  const height = OFFER_HEIGHT + (demos.length === 2 && columns === 1 ? STACKED_EXTRA_HEIGHT : 0);
  const heading = phase === 'on' ? say.onTitle : phase === 'declined' ? say.offTitle : say.title;
  const body = phase === 'on' ? say.onDescription : phase === 'declined' ? say.offDescription : say.description;

  return (
    <div
      role="group"
      aria-labelledby={titleId}
      style={{ ...offerPosition(toast.anchor, viewport, width, height), width, contain: 'layout paint' }}
      // Keep focus (and the caret) in the user's editor when a button on the card is pressed.
      onMouseDown={(event) => event.preventDefault()}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className="_echo_$_pointer-events-auto _echo_$_fixed _echo_$_flex _echo_$_overflow-hidden _echo_$_rounded-2xl _echo_$_bg-surface _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15 _echo_$_will-change-transform"
    >
      <span aria-hidden="true" className="_echo_$_w-1.5 _echo_$_shrink-0 _echo_$_bg-success" />
      <div className="_echo_$_min-w-0 _echo_$_flex-1">
        <div role="status">{header}</div>
        <div className="_echo_$_border-0 _echo_$_border-t _echo_$_border-solid _echo_$_border-sheen/15 _echo_$_px-4 _echo_$_pb-4 _echo_$_pt-3.5">
          <p id={titleId} className="_echo_$_m-0 _echo_$_font-sans _echo_$_text-[15px] _echo_$_font-semibold _echo_$_leading-[22px] _echo_$_text-fg">
            {heading}
          </p>
          <p className="_echo_$_m-0 _echo_$_mt-1 _echo_$_font-sans _echo_$_text-[14px] _echo_$_leading-5 _echo_$_text-fg-secondary">{body}</p>

          <div className={`_echo_$_mt-3.5 _echo_$_grid _echo_$_gap-2.5 ${columns === 2 ? '_echo_$_grid-cols-2' : '_echo_$_grid-cols-1'}`}>
            {offer.hold ? (
              <Demo label="Press and hold" hint="Hold on selected text for 1 second, then pick a style.">
                <HoldDemo />
              </Demo>
            ) : null}
            {offer.shortcuts ? (
              <Demo label="Keyboard shortcut" hint={`Select text, then press ${shortcutLabel('professional', offer.platform)} for Professional.`}>
                <ShortcutDemo keys={shortcutKeys('professional', offer.platform)} />
              </Demo>
            ) : null}
          </div>

          {phase === 'on' || phase === 'declined' ? null : (
            <div className="_echo_$_mt-4 _echo_$_flex _echo_$_flex-wrap _echo_$_items-center _echo_$_gap-x-3 _echo_$_gap-y-2">
              <button type="button" disabled={phase === 'asking'} onClick={allow} className={BUTTON}>
                {phase === 'asking' ? 'Waiting for Chrome…' : 'Allow on websites'}
              </button>
              <button type="button" disabled={phase === 'asking'} onClick={onDismiss} className={SECONDARY_BUTTON}>
                Not now
              </button>
              <span className="_echo_$_ml-auto _echo_$_flex _echo_$_items-center _echo_$_gap-1.5 _echo_$_font-sans _echo_$_text-[13px] _echo_$_leading-5 _echo_$_text-fg-soft">
                <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="_echo_$_shrink-0 _echo_$_text-accent">
                  <path d="M8 1.5 2.5 3.8v3.6c0 3.3 2.3 6.1 5.5 7.1 3.2-1 5.5-3.8 5.5-7.1V3.8L8 1.5Z" />
                  <path d="m5.6 8 1.7 1.7 3.2-3.3" />
                </svg>
                Secure, on-device processing
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** A looping picture over what it shows, in words. */
function Demo({ label, hint, children }: { label: string; hint: string; children: ReactNode }) {
  return (
    <div className="_echo_$_min-w-0">
      {children}
      <p className="_echo_$_m-0 _echo_$_mt-2 _echo_$_font-sans _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-fg">{label}</p>
      <p className="_echo_$_m-0 _echo_$_font-sans _echo_$_text-[13px] _echo_$_leading-5 _echo_$_text-fg-soft">{hint}</p>
    </div>
  );
}

/**
 * Press and hold: the words are selected, the pointer rests on them while a
 * ring fills for one second (the real hold), and the styles open beside it.
 * A loop of motion.duration.demo. Without animation (reduced motion) it
 * rests on the last picture: menu open, first style picked.
 */
function HoldDemo() {
  return (
    <div aria-hidden="true" className={STAGE}>
      <div className="_echo_$_absolute _echo_$_inset-0 _echo_$_animate-echo-demo-stage">
        <span className="_echo_$_absolute _echo_$_left-3.5 _echo_$_top-4 _echo_$_font-sans _echo_$_text-[13px] _echo_$_font-medium _echo_$_leading-5 _echo_$_text-fg">
          <span className="_echo_$_absolute _echo_$_-inset-x-1 _echo_$_inset-y-0 _echo_$_origin-left _echo_$_rounded-sm _echo_$_bg-accent/30 _echo_$_animate-echo-demo-select" />
          <span className="_echo_$_relative">Sorry for the delay.</span>
        </span>

        <div className="_echo_$_absolute _echo_$_left-[96px] _echo_$_top-[24px] _echo_$_z-10 _echo_$_translate-x-[46px] _echo_$_translate-y-[34px] _echo_$_animate-echo-demo-pointer">
          <svg width="28" height="28" viewBox="0 0 28 28" className="_echo_$_absolute _echo_$_-left-[14px] _echo_$_-top-[14px] _echo_$_text-accent _echo_$_opacity-0 _echo_$_animate-echo-demo-ring">
            <circle cx="14" cy="14" r="11" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeDasharray="69.115" strokeDashoffset="0" transform="rotate(-90 14 14)" />
          </svg>
          <svg width="16" height="16" viewBox="0 0 16 16" className="_echo_$_relative">
            <path d="M2 1.5v11l3-2.6 2 4.1 2-.9-2-4h4.2z" fill="#ffffff" stroke="#0b0b12" strokeWidth="1.2" strokeLinejoin="round" />
          </svg>
        </div>

        <div className="_echo_$_absolute _echo_$_left-[104px] _echo_$_top-[46px] _echo_$_w-[84px] _echo_$_overflow-hidden _echo_$_rounded-lg _echo_$_bg-surface _echo_$_p-1 _echo_$_shadow-card _echo_$_ring-1 _echo_$_ring-sheen/20 _echo_$_animate-echo-demo-menu">
          <span className="_echo_$_absolute _echo_$_inset-x-1 _echo_$_top-1 _echo_$_h-5 _echo_$_rounded-md _echo_$_bg-accent/35 _echo_$_animate-echo-demo-pick" />
          {(['professional', 'friendly', 'concise'] as const).map((style) => (
            <span key={style} className="_echo_$_relative _echo_$_flex _echo_$_h-5 _echo_$_items-center _echo_$_px-1.5 _echo_$_font-sans _echo_$_text-[11px] _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-fg">
              {styleLabel(style)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/**
 * The shortcut: the keys go down one after another, then a small rewrite
 * card shows up. Without animation it rests on the last picture: keys down, card shown.
 */
function ShortcutDemo({ keys }: { keys: readonly string[] }) {
  const pressed = ['_echo_$_animate-echo-demo-key-1', '_echo_$_animate-echo-demo-key-2', '_echo_$_animate-echo-demo-key-3'];
  return (
    <div aria-hidden="true" className={STAGE}>
      <div className="_echo_$_absolute _echo_$_inset-0 _echo_$_animate-echo-demo-stage">
        <div className="_echo_$_absolute _echo_$_inset-x-0 _echo_$_top-4 _echo_$_flex _echo_$_justify-center _echo_$_gap-1.5">
          {keys.map((cap, index) => (
            <span
              key={cap}
              className="_echo_$_relative _echo_$_flex _echo_$_h-8 _echo_$_min-w-8 _echo_$_shrink-0 _echo_$_items-center _echo_$_whitespace-nowrap _echo_$_justify-center _echo_$_rounded-lg _echo_$_border _echo_$_border-b-[3px] _echo_$_border-solid _echo_$_border-line-strong _echo_$_bg-surface _echo_$_px-2 _echo_$_font-sans _echo_$_text-[12px] _echo_$_font-semibold _echo_$_text-fg"
            >
              {cap}
              <span className={`_echo_$_absolute _echo_$_-inset-px _echo_$_flex _echo_$_translate-y-0.5 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg _echo_$_bg-accent _echo_$_text-on-accent ${pressed[index] ?? ''}`}>{cap}</span>
            </span>
          ))}
        </div>
        <div className="_echo_$_absolute _echo_$_inset-x-3 _echo_$_top-[62px] _echo_$_rounded-lg _echo_$_bg-surface _echo_$_p-2 _echo_$_shadow-card _echo_$_ring-1 _echo_$_ring-sheen/20 _echo_$_animate-echo-demo-chip">
          <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-1.5 _echo_$_font-sans _echo_$_text-[11px] _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-fg">
            <Logo size={14} />
            Rewrite ready
          </div>
          <div className="_echo_$_mt-1.5 _echo_$_space-y-1">
            <span className="_echo_$_block _echo_$_h-1.5 _echo_$_w-full _echo_$_rounded-full _echo_$_bg-sheen/25" />
            <span className="_echo_$_block _echo_$_h-1.5 _echo_$_w-2/3 _echo_$_rounded-full _echo_$_bg-sheen/25" />
          </div>
        </div>
      </div>
    </div>
  );
}
