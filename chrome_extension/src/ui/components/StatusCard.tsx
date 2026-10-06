import { formatBytes } from '../../shared/format';
import type { EngineStatus } from '../../shared/status';
import { EchoLoader } from '../brand/EchoLoader';
import { PRIMARY_BUTTON } from './buttons';

interface Presentation {
  readonly label: string;
  readonly dot: string;
  readonly detail: string;
  readonly action?: string;
}

export function presentStatus(status: EngineStatus): Presentation {
  switch (status.state) {
    case 'unloaded':
      return {
        label: 'Sleeping',
        dot: '_echo_$_bg-ink-300',
        detail: 'Wakes up on your first rewrite, or load it now to skip the wait.',
        action: 'Load now',
      };
    case 'loading':
      return {
        label: `Loading · ${Math.round(status.progress * 100)}%`,
        dot: '_echo_$_bg-echo-500 _echo_$_animate-pulse',
        detail: 'Reading the model into memory. Only the first rewrite waits for this.',
      };
    case 'ready':
      return {
        label: 'Ready',
        dot: '_echo_$_bg-better-500',
        detail: `${status.model.displayName} · ${status.model.precision} · ${formatBytes(status.model.sizeBytes)} · on-device`,
      };
    case 'error':
      return { label: "Couldn't load", dot: '_echo_$_bg-danger-500', detail: status.message, action: 'Try again' };
  }
}

export function StatusCard({ status, onLoad }: { status: EngineStatus; onLoad: () => void }) {
  const view = presentStatus(status);
  return (
    <section
      aria-label="Writing model status"
      className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_p-4 _echo_$_shadow-card dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900"
    >
      <div className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
        <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-2">
          {status.state === 'loading' ? (
            <EchoLoader size={16} label="Loading" />
          ) : (
            <span aria-hidden="true" className={`_echo_$_h-2 _echo_$_w-2 _echo_$_rounded-full ${view.dot}`} />
          )}
          <span className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">{view.label}</span>
        </div>
        {view.action ? (
          <button
            type="button"
            onClick={onLoad}
            className={PRIMARY_BUTTON}
          >
            {view.action}
          </button>
        ) : null}
      </div>
      {status.state === 'loading' ? (
        <div
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(status.progress * 100)}
          className="_echo_$_mt-3 _echo_$_h-1.5 _echo_$_overflow-hidden _echo_$_rounded-full _echo_$_bg-ink-100 dark:_echo_$_bg-ink-800"
        >
          <div className="_echo_$_h-full _echo_$_rounded-full _echo_$_bg-echo-gradient _echo_$_transition-[width] _echo_$_duration-300" style={{ width: `${Math.max(4, status.progress * 100)}%` }} />
        </div>
      ) : null}
      <p className="_echo_$_mt-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{view.detail}</p>
    </section>
  );
}
