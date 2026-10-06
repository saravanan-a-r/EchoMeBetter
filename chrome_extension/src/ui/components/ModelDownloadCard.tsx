/**
 * Shown instead of the model status while the model is not on this device:
 * the one-time download, its progress, and what to do when it stops.
 */
import { useState } from 'react';
import { describeError } from '../../shared/errors';
import { formatBytes, formatTimeLeft } from '../../shared/format';
import type { DownloadState } from '../../shared/modelInstall';
import { EchoLoader } from '../brand/EchoLoader';
import { DANGER_BUTTON, PRIMARY_BUTTON, SECONDARY_BUTTON } from './buttons';
import { ModelSourceUrl } from './ModelSourceUrl';

export interface ModelDownloadCardProps {
  readonly download: DownloadState;
  readonly onDownload: () => void;
  readonly onCancel: () => void;
  /** A short confirmation shown above the card's text, e.g. right after the model was removed. */
  readonly notice?: string;
}

interface Presentation {
  readonly label: string;
  readonly detail: string;
  /** 0-100 while a download runs. */
  readonly percent?: number;
}

export function presentDownload(download: DownloadState): Presentation {
  switch (download.state) {
    case 'idle':
      return {
        label: 'Writing model needed',
        detail:
          "EchoMeBetter needs its writing model before it can rewrite text. It's a one-time download, stored privately on this device and shared by every website.",
      };
    case 'failed':
      return { label: 'Download stopped', detail: describeError(download.error) };
    case 'downloading': {
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
  }
}

function DownloadGlyph() {
  return (
    <span aria-hidden="true" className="_echo_$_flex _echo_$_h-5 _echo_$_w-5 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-full _echo_$_bg-echo-50 _echo_$_text-echo-600 dark:_echo_$_bg-echo-950 dark:_echo_$_text-echo-300">
      <svg width="12" height="12" viewBox="0 0 12 12">
        <path d="M6 1.5v6M3.5 5 6 7.5 8.5 5M2.5 10h7" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

export function ModelDownloadCard({ download, onDownload, onCancel, notice }: ModelDownloadCardProps) {
  const [confirmingCancel, setConfirmingCancel] = useState(false);
  const running = download.state === 'downloading';
  // Once the download has ended there is nothing left to confirm.
  const confirming = confirmingCancel && running;
  const view = presentDownload(download);

  return (
    <section
      aria-label="Writing model download"
      className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_p-4 _echo_$_shadow-card dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900"
    >
      {notice && download.state === 'idle' ? (
        <p role="status" className="_echo_$_mb-3 _echo_$_rounded-lg _echo_$_bg-better-200/40 _echo_$_px-2.5 _echo_$_py-1.5 _echo_$_text-xs _echo_$_font-medium _echo_$_text-ink-800 dark:_echo_$_bg-better-600/20 dark:_echo_$_text-better-200">
          {notice}
        </p>
      ) : null}

      <div className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
        <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-2">
          {running ? (
            <EchoLoader size={16} label="Downloading" />
          ) : download.state === 'failed' ? (
            <span aria-hidden="true" className="_echo_$_h-2 _echo_$_w-2 _echo_$_rounded-full _echo_$_bg-danger-500" />
          ) : (
            <DownloadGlyph />
          )}
          <span className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">{view.label}</span>
        </div>
        {download.state === 'failed' ? (
          <button type="button" onClick={onDownload} className={PRIMARY_BUTTON}>
            Try again
          </button>
        ) : null}
      </div>

      {view.percent !== undefined ? (
        <div
          role="progressbar"
          aria-label="Download progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={view.percent}
          className="_echo_$_mt-3 _echo_$_h-1.5 _echo_$_overflow-hidden _echo_$_rounded-full _echo_$_bg-ink-100 dark:_echo_$_bg-ink-800"
        >
          <div
            className={`_echo_$_h-full _echo_$_rounded-full _echo_$_bg-echo-gradient _echo_$_transition-[width] _echo_$_duration-300 ${
              download.state === 'downloading' && download.phase === 'verifying' ? '_echo_$_animate-pulse' : ''
            }`}
            style={{ width: `${Math.max(4, view.percent)}%` }}
          />
        </div>
      ) : null}

      <p className="_echo_$_mt-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{view.detail}</p>
      {download.state === 'idle' ? (
        <button type="button" onClick={onDownload} className={`${PRIMARY_BUTTON} _echo_$_mt-3 _echo_$_w-full`}>
          Download model
        </button>
      ) : null}
      {running && !confirming ? (
        <div className="_echo_$_mt-2 _echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
          <p className="_echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-400">You can close this window. The download keeps going.</p>
          <button type="button" onClick={() => setConfirmingCancel(true)} className={SECONDARY_BUTTON}>
            Cancel download
          </button>
        </div>
      ) : null}

      {confirming ? (
        <div
          role="group"
          aria-labelledby="cancel-download-title"
          className="_echo_$_mt-3 _echo_$_rounded-xl _echo_$_border _echo_$_border-danger-400/40 _echo_$_bg-danger-50 _echo_$_p-3 dark:_echo_$_bg-danger-950"
        >
          <p id="cancel-download-title" className="_echo_$_text-sm _echo_$_font-medium _echo_$_text-ink-900 dark:_echo_$_text-white">
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
