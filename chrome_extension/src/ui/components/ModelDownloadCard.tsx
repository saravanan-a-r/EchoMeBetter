/**
 * A download in progress, or one that stopped: of the writing model with
 * the first styles, or of styles added later. Shown in place of the model
 * status until it is done or put aside.
 */
import { useState } from 'react';
import { describeError } from '../../shared/errors';
import { formatBytes, formatList, formatTimeLeft } from '../../shared/format';
import type { DownloadState, DownloadTarget } from '../../shared/modelInstall';
import { styleLabel } from '../../shared/styles';
import { EchoLoader } from '../brand/EchoLoader';
import { DANGER_BUTTON, PRIMARY_BUTTON, SECONDARY_BUTTON } from './buttons';
import { ModelSourceUrl } from './ModelSourceUrl';

export interface ModelDownloadCardProps {
  readonly download: Exclude<DownloadState, { state: 'idle' }>;
  /** Start the same download again. */
  readonly onRetry: (target: DownloadTarget) => void;
  /** Stop a running download, or put a failed one aside. */
  readonly onCancel: () => void;
}

interface Presentation {
  readonly label: string;
  readonly detail: string;
  /** 0-100 while a download runs. */
  readonly percent?: number;
}

/** "Writing model and Professional style", "Professional and Grammar styles". */
export function describeTarget(target: DownloadTarget): string {
  const styles = target.adapters.map(styleLabel);
  const adapters = styles.length === 0 ? '' : `${formatList(styles)} ${styles.length === 1 ? 'style' : 'styles'}`;
  if (!target.base) return adapters;
  return adapters ? `Writing model and ${adapters}` : 'Writing model';
}

export function presentDownload(download: ModelDownloadCardProps['download']): Presentation {
  if (download.state === 'failed') return { label: 'Download stopped', detail: describeError(download.error) };
  if (download.phase === 'verifying') {
    return { label: 'Checking files…', detail: 'Making sure every file arrived intact.', percent: 100 };
  }
  if (download.totalBytes === 0) return { label: 'Downloading', detail: 'Connecting…', percent: 0 };
  const percent = Math.min(100, Math.floor((download.receivedBytes / download.totalBytes) * 100));
  const parts = [`${formatBytes(download.receivedBytes)} of ${formatBytes(download.totalBytes)}`];
  if (download.bytesPerSecond > 0) {
    parts.push(`${formatBytes(download.bytesPerSecond)}/s`);
    parts.push(formatTimeLeft((download.totalBytes - download.receivedBytes) / download.bytesPerSecond));
  }
  return { label: `Downloading · ${percent}%`, detail: parts.join(' · '), percent };
}

export function ModelDownloadCard({ download, onRetry, onCancel }: ModelDownloadCardProps) {
  const [confirmingCancel, setConfirmingCancel] = useState(false);
  const running = download.state === 'downloading';
  // Once the download has ended there is nothing left to confirm.
  const confirming = confirmingCancel && running;
  const view = presentDownload(download);

  return (
    <section
      aria-label="Download"
      className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_p-4 _echo_$_shadow-card"
    >
      <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-2">
        {running ? (
          <EchoLoader size={16} label="Downloading" />
        ) : (
          <span aria-hidden="true" className="_echo_$_h-2 _echo_$_w-2 _echo_$_rounded-full _echo_$_bg-danger" />
        )}
        <span className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-fg">{view.label}</span>
      </div>
      <p className="_echo_$_mt-1 _echo_$_text-xs _echo_$_font-medium _echo_$_text-fg-secondary">{describeTarget(download.target)}</p>

      {view.percent !== undefined ? (
        <div
          role="progressbar"
          aria-label="Download progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={view.percent}
          className="_echo_$_mt-3 _echo_$_h-1.5 _echo_$_overflow-hidden _echo_$_rounded-full _echo_$_bg-muted"
        >
          <div
            className={`_echo_$_h-full _echo_$_rounded-full _echo_$_bg-echo-gradient _echo_$_transition-[width] _echo_$_duration-300 ${
              download.state === 'downloading' && download.phase === 'verifying' ? '_echo_$_animate-pulse' : ''
            }`}
            style={{ width: `${Math.max(4, view.percent)}%` }}
          />
        </div>
      ) : null}

      <p className="_echo_$_mt-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">{view.detail}</p>

      {download.state === 'failed' ? (
        <div className="_echo_$_mt-3 _echo_$_flex _echo_$_justify-end _echo_$_gap-2">
          <button type="button" onClick={onCancel} className={SECONDARY_BUTTON}>
            Dismiss
          </button>
          <button type="button" onClick={() => onRetry(download.target)} className={PRIMARY_BUTTON}>
            Try again
          </button>
        </div>
      ) : null}

      {running && !confirming ? (
        <div className="_echo_$_mt-2 _echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
          <p className="_echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-subtle">You can close this window. The download keeps going.</p>
          <button type="button" onClick={() => setConfirmingCancel(true)} className={SECONDARY_BUTTON}>
            Cancel download
          </button>
        </div>
      ) : null}

      {confirming ? (
        <div
          role="group"
          aria-labelledby="cancel-download-title"
          className="_echo_$_mt-3 _echo_$_rounded-xl _echo_$_border _echo_$_border-danger-line/40 _echo_$_bg-danger-subtle _echo_$_p-3"
        >
          <p id="cancel-download-title" className="_echo_$_text-sm _echo_$_font-medium _echo_$_text-fg">
            Stop and delete the partly downloaded files?
          </p>
          <div className="_echo_$_mt-2.5 _echo_$_flex _echo_$_justify-end _echo_$_gap-2">
            <button type="button" onClick={() => setConfirmingCancel(false)} className={SECONDARY_BUTTON}>
              Keep downloading
            </button>
            <button
              type="button"
              onClick={() => {
                setConfirmingCancel(false);
                onCancel();
              }}
              className={DANGER_BUTTON}
            >
              Stop download
            </button>
          </div>
        </div>
      ) : null}

      <ModelSourceUrl />
    </section>
  );
}
