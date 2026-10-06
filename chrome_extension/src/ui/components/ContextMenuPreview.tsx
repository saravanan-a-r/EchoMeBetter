/** A drawing of the right-click menu, so the onboarding shows exactly what to look for. */
import { shortcutLabel, type KeyPlatform } from '../../shared/shortcuts';
import { STYLES } from '../../shared/styles';
import { Logo } from '../brand/Logo';

/** `shortcuts`: draw each style's shortcut next to it, as the real menu shows it while shortcuts are on. */
export function ContextMenuPreview({ shortcuts }: { shortcuts?: KeyPlatform }) {
  const item = '_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_px-3 _echo_$_py-1.5 _echo_$_text-[13px]';
  return (
    <div aria-hidden="true" className="_echo_$_relative _echo_$_flex _echo_$_select-none _echo_$_items-start _echo_$_gap-1 _echo_$_font-sans">
      <div className="_echo_$_w-48 _echo_$_rounded-lg _echo_$_border _echo_$_border-ink-200 _echo_$_bg-white _echo_$_py-1 _echo_$_text-ink-800 _echo_$_shadow-card dark:_echo_$_border-ink-700 dark:_echo_$_bg-ink-800 dark:_echo_$_text-ink-100">
        <div className={item}>Cut</div>
        <div className={item}>Copy</div>
        <div className={item}>Paste</div>
        <div className="_echo_$_my-1 _echo_$_h-px _echo_$_bg-ink-100 dark:_echo_$_bg-ink-700" />
        <div className={`${item} _echo_$_bg-echo-50 _echo_$_font-medium _echo_$_text-echo-700 dark:_echo_$_bg-echo-950 dark:_echo_$_text-echo-200`}>
          <span className="_echo_$_flex _echo_$_items-center _echo_$_gap-2">
            <Logo size={14} /> EchoMeBetter
          </span>
          <span>›</span>
        </div>
        <div className={item}>Inspect</div>
      </div>
      <div className={`_echo_$_mt-[100px] ${shortcuts ? '_echo_$_w-60' : '_echo_$_w-40'} _echo_$_rounded-lg _echo_$_border _echo_$_border-ink-200 _echo_$_bg-white _echo_$_py-1 _echo_$_text-ink-800 _echo_$_shadow-card dark:_echo_$_border-ink-700 dark:_echo_$_bg-ink-800 dark:_echo_$_text-ink-100`}>
        {STYLES.map((style, index) => (
          <div key={style.id} className={`${item} ${index === 0 ? '_echo_$_bg-echo-600 _echo_$_text-white' : ''}`}>
            {style.label}
            {shortcuts ? <span className={index === 0 ? '_echo_$_text-white/80' : '_echo_$_text-ink-400'}>{shortcutLabel(style.id, shortcuts)}</span> : null}
          </div>
        ))}
      </div>
    </div>
  );
}
