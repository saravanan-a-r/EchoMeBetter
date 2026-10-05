/**
 * The EchoMeBetter mark: a voice (the dot) and its echo (two arcs), with a
 * spark for "better", on the violet-to-mint brand tile.
 *
 * The geometry is shared with EchoLoader so the loader reads as the logo
 * coming alive, and with the toolbar icons rendered from it.
 */
import { useId } from 'react';

export const MARK = {
  dot: { cx: 20, cy: 32, r: 5 },
  innerArc: 'M27.7 22.8 A12 12 0 0 1 27.7 41.2',
  outerArc: 'M33.5 15.9 A21 21 0 0 1 33.5 48.1',
  spark: 'M48 9 C48.6 13.6 50.4 15.4 55 16 C50.4 16.6 48.6 18.4 48 23 C47.4 18.4 45.6 16.6 41 16 C45.6 15.4 47.4 13.6 48 9Z',
} as const;

export function Logo({ size = 32, title = 'EchoMeBetter' }: { size?: number; title?: string }) {
  const gradient = useId();
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" role="img" aria-label={title}>
      <defs>
        <linearGradient id={gradient} x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#7650FF" />
          <stop offset="0.55" stopColor="#5B6CFF" />
          <stop offset="1" stopColor="#2DD4BF" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="16" fill={`url(#${gradient})`} />
      <circle {...MARK.dot} fill="#fff" />
      <path d={MARK.innerArc} stroke="#fff" strokeWidth="4.5" strokeLinecap="round" fill="none" />
      <path d={MARK.outerArc} stroke="#fff" strokeOpacity="0.85" strokeWidth="4.5" strokeLinecap="round" fill="none" />
      <path d={MARK.spark} fill="#E6FFFA" />
    </svg>
  );
}

export function Wordmark({ className = '' }: { className?: string }) {
  return (
    <span className={`_echo_$_font-semibold _echo_$_tracking-tight ${className}`}>
      Echo<span className="_echo_$_text-echo-500">Me</span>Better
    </span>
  );
}
