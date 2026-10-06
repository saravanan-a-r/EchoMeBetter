/**
 * Shortcuts are on but websites are not allowed yet: ask, and say plainly
 * what the permission is for.
 */
import { shortcutLabel, type KeyPlatform } from '../../shared/shortcuts';
import { PRIMARY_BUTTON, SECONDARY_BUTTON } from './buttons';

export const ACCESS_TITLE = 'Turn on keyboard shortcuts';

export function accessExplanation(platform: KeyPlatform): string {
  return `To react to shortcuts like ${shortcutLabel('professional', platform)} on websites, EchoMeBetter needs your OK to run on them. Chrome will ask you to confirm. Your text still never leaves this device.`;
}

export function ShortcutAccessCard({
  platform,
  onAllow,
  onNotNow,
  title = ACCESS_TITLE,
}: {
  platform: KeyPlatform;
  onAllow: () => void;
  onNotNow?: () => void;
  /** Where shortcuts are already switched on, the card is the remaining step rather than the invitation. */
  title?: string;
}) {
  return (
    <section
      aria-labelledby="shortcut-access-title"
      className="_echo_$_rounded-xl _echo_$_border _echo_$_border-echo-200 _echo_$_bg-echo-50 _echo_$_p-3 dark:_echo_$_border-echo-800 dark:_echo_$_bg-echo-950"
    >
      <h3 id="shortcut-access-title" className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">
        {title}
      </h3>
      <p className="_echo_$_mt-1 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-600 dark:_echo_$_text-ink-200">{accessExplanation(platform)}</p>
      <div className="_echo_$_mt-2.5 _echo_$_flex _echo_$_justify-end _echo_$_gap-2">
        {onNotNow ? (
          <button type="button" onClick={onNotNow} className={SECONDARY_BUTTON}>
            Not now
          </button>
        ) : null}
        <button type="button" onClick={onAllow} className={PRIMARY_BUTTON}>
          Allow on websites
        </button>
      </div>
    </section>
  );
}
