/**
 * "Remove downloaded model": frees the disk space the model takes. Asks
 * first, in place, and says plainly what removing it means.
 */
import { useState } from 'react';
import { formatBytes } from '../../shared/format';
import type { UiReply } from '../../shared/messages';
import { DANGER_BUTTON, SECONDARY_BUTTON } from './buttons';

export interface RemoveModelProps {
  readonly sizeBytes: number;
  /** Resolves once the request is accepted; the model's files then go away shortly after. */
  readonly onRemove: () => Promise<UiReply>;
}

export function RemoveModel({ sizeBytes, onRemove }: RemoveModelProps) {
  const [step, setStep] = useState<'idle' | 'confirm' | 'removing'>('idle');
  const [problem, setProblem] = useState<string | null>(null);

  if (step === 'idle') {
    return (
      <div className="_echo_$_flex _echo_$_justify-end">
        <button
          type="button"
          onClick={() => {
            setProblem(null);
            setStep('confirm');
          }}
          className="_echo_$_rounded _echo_$_text-xs _echo_$_font-medium _echo_$_text-ink-500 _echo_$_underline-offset-2 hover:_echo_$_text-danger-500 hover:_echo_$_underline focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-echo-500 dark:_echo_$_text-ink-300"
        >
          Remove downloaded model
        </button>
      </div>
    );
  }

  const remove = async () => {
    setStep('removing');
    setProblem(null);
    const reply = await onRemove().catch((): UiReply => ({ ok: false, error: { code: 'INTERNAL' } }));
    if (reply.ok) return;
    setProblem(reply.error.code === 'BUSY' ? 'Wait for the current rewrite to finish.' : "Couldn't remove the model. Please try again.");
    setStep('confirm');
  };

  return (
    <section
      role="group"
      aria-labelledby="remove-model-title"
      className="_echo_$_rounded-xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_p-3.5 dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900"
    >
      <h2 id="remove-model-title" className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">
        Remove the model from this device?
      </h2>
      <p className="_echo_$_mt-1 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
        This frees about {formatBytes(sizeBytes)} of disk space. EchoMeBetter can't rewrite text until you download the model again. Your settings are
        kept.
      </p>
      {problem ? (
        <p role="alert" className="_echo_$_mt-2 _echo_$_text-xs _echo_$_font-medium _echo_$_text-danger-500">
          {problem}
        </p>
      ) : null}
      <div className="_echo_$_mt-3 _echo_$_flex _echo_$_justify-end _echo_$_gap-2">
        <button type="button" disabled={step === 'removing'} onClick={() => setStep('idle')} className={SECONDARY_BUTTON}>
          Keep it
        </button>
        <button type="button" disabled={step === 'removing'} onClick={() => void remove()} className={DANGER_BUTTON}>
          {step === 'removing' ? 'Removing…' : 'Remove model'}
        </button>
      </div>
    </section>
  );
}
