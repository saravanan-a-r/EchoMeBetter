/**
 * The popup's styles, one list in menu order whatever their state. Each row
 * says what the style does; at its end, a style that is not on this device
 * shows what it takes (its size) beside a download icon, and one that is
 * shows a check. A style that runs on another's download says so instead.
 */
import { useId } from 'react';
import { formatBytes } from '../../shared/format';
import type { StyleReadiness } from '../../shared/modelInstall';
import { STYLES, styleLabel, type StyleDescriptor, type StyleId } from '../../shared/styles';
import { StyleGlyph } from '../brand/StyleGlyph';
import { StyleStatus } from './StyleStatus';

export interface PopupStylesProps {
  /** Per style: ready, or what it still needs. */
  readonly readiness: Readonly<Partial<Record<StyleId, StyleReadiness>>>;
  /** A download is running, so no other can start. */
  readonly busy: boolean;
  /** The styles whose download is running now. */
  readonly downloading: readonly StyleId[];
  readonly onDownload: (adapter: StyleId) => void;
}

/** What the row's end says about size or dependence, or null when there is nothing to add. */
function sizeNote(style: StyleDescriptor, readiness: StyleReadiness | undefined): string | null {
  if (readiness?.state !== 'needs-download' && readiness?.state !== 'needs-model') return null;
  return readiness.adapter.style === style.id ? formatBytes(readiness.adapter.sizeBytes) : `Runs on ${styleLabel(readiness.adapter.style)}`;
}

export function PopupStyles({ readiness, busy, downloading, onDownload }: PopupStylesProps) {
  const titleId = useId();
  const installed = STYLES.filter((style) => readiness[style.id]?.state === 'ready').length;
  const known = STYLES.some((style) => readiness[style.id] !== undefined);
  return (
    <section aria-labelledby={titleId}>
      <h2 id={titleId} className="_echo_$_mb-2 _echo_$_text-[11px] _echo_$_font-semibold _echo_$_uppercase _echo_$_tracking-wider _echo_$_text-fg-muted">
        Styles
      </h2>
      {known && installed === 0 ? <p className="_echo_$_mb-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">Download a style to start rewriting.</p> : null}
      {/* No overflow clipping: the status icons' tooltips open above their rows. */}
      <ul className="_echo_$_divide-y _echo_$_divide-line _echo_$_rounded-xl _echo_$_border _echo_$_border-line _echo_$_bg-surface">
        {STYLES.map((style) => {
          const note = sizeNote(style, readiness[style.id]);
          return (
            <li key={style.id} className="_echo_$_flex _echo_$_items-center _echo_$_gap-2.5 _echo_$_px-3 _echo_$_py-1.5">
              <span className="_echo_$_flex _echo_$_h-6 _echo_$_w-6 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-md _echo_$_bg-accent-subtle _echo_$_text-accent-fg">
                <StyleGlyph style={style.id} size={14} />
              </span>
              <span className="_echo_$_min-w-0 _echo_$_flex-1 _echo_$_leading-4">
                <span className="_echo_$_block _echo_$_text-sm _echo_$_font-semibold _echo_$_text-fg">{style.label}</span>
                <span className="_echo_$_block _echo_$_truncate _echo_$_text-[11px] _echo_$_text-fg-muted">{style.hint}</span>
              </span>
              <span className="_echo_$_ml-auto _echo_$_flex _echo_$_shrink-0 _echo_$_items-center _echo_$_gap-1.5">
                {note ? <span className="_echo_$_text-xs _echo_$_tabular-nums _echo_$_text-fg-muted">{note}</span> : null}
                <StyleStatus style={style} readiness={readiness[style.id]} downloading={downloading} busy={busy} onDownload={onDownload} />
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
