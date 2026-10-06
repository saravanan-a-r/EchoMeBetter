/**
 * Development aid: shows the URL the model is downloaded from.
 *
 * Temporary. Remove this file and its one use in ModelDownloadCard before
 * the extension is published.
 */
import { MODEL_SOURCE_URL } from '../../shared/modelSource';

export function ModelSourceUrl() {
  return (
    <p className="_echo_$_mt-3 _echo_$_flex _echo_$_items-baseline _echo_$_gap-2 _echo_$_rounded-lg _echo_$_bg-ink-50 _echo_$_px-2.5 _echo_$_py-1.5 _echo_$_text-[11px] _echo_$_text-ink-500 dark:_echo_$_bg-ink-950 dark:_echo_$_text-ink-300">
      <span className="_echo_$_shrink-0 _echo_$_font-semibold _echo_$_uppercase _echo_$_tracking-wider">Source</span>
      <span className="_echo_$_min-w-0 _echo_$_break-all _echo_$_font-mono">{MODEL_SOURCE_URL}</span>
    </p>
  );
}
