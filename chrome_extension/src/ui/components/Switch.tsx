/** An on/off switch; `labelledBy` points at the visible label. */
export function Switch({ checked, onChange, labelledBy, describedBy }: { checked: boolean; onChange: (next: boolean) => void; labelledBy: string; describedBy?: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-labelledby={labelledBy}
      aria-describedby={describedBy}
      onClick={() => onChange(!checked)}
      className={`_echo_$_relative _echo_$_inline-flex _echo_$_h-6 _echo_$_w-11 _echo_$_shrink-0 _echo_$_cursor-pointer _echo_$_items-center _echo_$_rounded-full _echo_$_transition-colors focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-offset-2 focus-visible:_echo_$_outline-focus ${
        checked ? '_echo_$_bg-accent' : '_echo_$_bg-control-off'
      }`}
    >
      <span
        aria-hidden="true"
        className={`_echo_$_inline-block _echo_$_h-5 _echo_$_w-5 _echo_$_rounded-full _echo_$_bg-control-knob _echo_$_shadow _echo_$_transition-transform ${checked ? '_echo_$_translate-x-[22px]' : '_echo_$_translate-x-0.5'}`}
      />
    </button>
  );
}
