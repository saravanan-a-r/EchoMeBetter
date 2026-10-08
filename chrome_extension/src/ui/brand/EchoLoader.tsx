/**
 * The brand loader: the logo's voice and echoes pulsing outward in turn.
 * Pure CSS animation (transform/opacity only), so it costs the page nothing
 * but compositing.
 */
import { useId } from 'react';
import { brand } from '../../design/tokens.cjs';
import { MARK } from './Logo';

export function EchoLoader({ size = 24, label = 'Working' }: { size?: number; label?: string }) {
  const gradient = useId();
  const wave = '_echo_$_animate-echo-wave _echo_$_[transform-box:fill-box] _echo_$_[transform-origin:left_center]';
  return (
    <svg width={size} height={size} viewBox="8 6 52 52" role="img" aria-label={label}>
      <defs>
        <linearGradient id={gradient} x1="8" y1="6" x2="60" y2="58" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor={brand['loader-start']} />
          <stop offset="1" stopColor={brand['loader-end']} />
        </linearGradient>
      </defs>
      <circle {...MARK.dot} fill={`url(#${gradient})`} className={wave} />
      <path
        d={MARK.innerArc}
        stroke={`url(#${gradient})`}
        strokeWidth="4.5"
        strokeLinecap="round"
        fill="none"
        className={`${wave} _echo_$_[animation-delay:150ms]`}
      />
      <path
        d={MARK.outerArc}
        stroke={`url(#${gradient})`}
        strokeWidth="4.5"
        strokeLinecap="round"
        fill="none"
        className={`${wave} _echo_$_[animation-delay:300ms]`}
      />
      <path d={MARK.spark} fill={brand['loader-end']} className={`${wave} _echo_$_[animation-delay:450ms]`} />
    </svg>
  );
}
