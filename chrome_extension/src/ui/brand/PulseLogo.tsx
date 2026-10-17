import { Logo } from './Logo';

/**
 * The logo with its echo: two faint rings spread from it in turn over a softly
 * breathing glow. Slow and low-contrast on purpose, and still for anyone who
 * prefers reduced motion.
 *
 *   mint    on a dark surface (the welcome page's hero)
 *   light   on the brand gradient (the popup's header): white rings and glow
 *
 * The rings keep the tile's proportions at any `size`.
 */
export function PulseLogo({ size, tone = 'mint' }: { size: number; tone?: 'mint' | 'light' }) {
  const ring = `_echo_$_absolute _echo_$_border _echo_$_animate-echo-ripple _echo_$_opacity-0 motion-reduce:_echo_$_hidden ${
    tone === 'mint' ? '_echo_$_border-better-300/50' : '_echo_$_border-white/60'
  }`;
  const ringStyle = { inset: (size * 2) / 72, borderRadius: (size * 18) / 72 };
  const glowReach = size / 3;
  return (
    <div className="_echo_$_relative _echo_$_flex _echo_$_items-center _echo_$_justify-center" style={{ width: size, height: size }}>
      <span
        aria-hidden="true"
        style={{ inset: -glowReach }}
        className={`_echo_$_absolute _echo_$_rounded-full _echo_$_blur-2xl _echo_$_animate-echo-glow-breath motion-reduce:_echo_$_hidden ${tone === 'mint' ? '_echo_$_bg-echo-500/25' : '_echo_$_bg-white/20'}`}
      />
      <span aria-hidden="true" className={ring} style={ringStyle} />
      <span aria-hidden="true" className={ring} style={{ ...ringStyle, animationDelay: '2500ms' }} />
      <span className="_echo_$_relative">
        <Logo size={size} />
      </span>
    </div>
  );
}
