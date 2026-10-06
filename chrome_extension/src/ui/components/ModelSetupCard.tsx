/**
 * First run, nothing on this device yet: what can be downloaded -- the
 * writing model every style needs, and the styles -- with the chosen parts
 * downloaded in one go. Every style is chosen to begin with; styles can also
 * be added (or removed) later.
 */
import { useState, type ReactNode } from 'react';
import { formatBytes, formatList } from '../../shared/format';
import { stylesUsing, type ModelCatalog } from '../../shared/modelInstall';
import { STYLE_IDS, styleLabel, type StyleId } from '../../shared/styles';
import { EchoLoader } from '../brand/EchoLoader';
import type { CatalogState } from '../hooks/useModel';
import { PRIMARY_BUTTON, SECONDARY_BUTTON } from './buttons';
import { ModelSourceUrl } from './ModelSourceUrl';

export interface ModelSetupCardProps {
  readonly catalog: CatalogState;
  readonly onDownload: (adapters: StyleId[]) => void;
  readonly onRetryCatalog: () => void;
  /** A short confirmation shown above the card's text, e.g. right after the model was removed. */
  readonly notice?: string;
}

const SETUP_DETAIL =
  'EchoMeBetter rewrites on your device. Download its writing model and the styles you want: a one-time download, stored privately on this device and shared by every website.';

/** "Also used by Grammar and Concise for now", when other styles borrow this one's adapter. */
function borrowedBy(catalog: ModelCatalog, adapter: StyleId): string | null {
  const others = stylesUsing(catalog, adapter).filter((style) => style !== adapter);
  return others.length > 0 ? `Also used by ${formatList(others.map(styleLabel))} for now` : null;
}

const CHECKBOX = '_echo_$_mt-0.5 _echo_$_h-4 _echo_$_w-4 _echo_$_shrink-0 _echo_$_accent-echo-600';

function Choice({ id, label, sizeBytes, note, children }: { id: string; label: string; sizeBytes: number; note?: string | null; children: ReactNode }) {
  return (
    <li className="_echo_$_flex _echo_$_items-start _echo_$_gap-2.5 _echo_$_px-3 _echo_$_py-2">
      {children}
      <label htmlFor={id} className="_echo_$_min-w-0 _echo_$_flex-1 _echo_$_text-sm">
        <span className="_echo_$_flex _echo_$_items-baseline _echo_$_justify-between _echo_$_gap-2">
          <span className="_echo_$_font-medium _echo_$_text-ink-900 dark:_echo_$_text-white">{label}</span>
          <span className="_echo_$_shrink-0 _echo_$_text-xs _echo_$_tabular-nums _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{formatBytes(sizeBytes)}</span>
        </span>
        {note ? <span className="_echo_$_block _echo_$_text-xs _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{note}</span> : null}
      </label>
    </li>
  );
}

function Choices({ catalog, onDownload }: { catalog: ModelCatalog; onDownload: (adapters: StyleId[]) => void }) {
  const [unchecked, setUnchecked] = useState<ReadonlySet<StyleId>>(new Set());
  const adapters = STYLE_IDS.flatMap((style) => catalog.adapters.filter((adapter) => adapter.style === style));
  const chosen = adapters.filter((adapter) => !unchecked.has(adapter.style));
  const total = catalog.model.sizeBytes + chosen.reduce((sum, adapter) => sum + adapter.sizeBytes, 0);
  const toggle = (style: StyleId) =>
    setUnchecked((current) => {
      const next = new Set(current);
      if (!next.delete(style)) next.add(style);
      return next;
    });

  return (
    <>
      <ul
        aria-label="What to download"
        className="_echo_$_mt-3 _echo_$_divide-y _echo_$_divide-ink-100 _echo_$_rounded-xl _echo_$_border _echo_$_border-ink-100 dark:_echo_$_divide-ink-800 dark:_echo_$_border-ink-800"
      >
        <Choice id="setup-model" label="Writing model" sizeBytes={catalog.model.sizeBytes} note="Required by every style">
          <input id="setup-model" type="checkbox" checked disabled className={CHECKBOX} />
        </Choice>
        {adapters.map((adapter) => (
          <Choice key={adapter.style} id={`setup-${adapter.style}`} label={`${styleLabel(adapter.style)} style`} sizeBytes={adapter.sizeBytes} note={borrowedBy(catalog, adapter.style)}>
            <input id={`setup-${adapter.style}`} type="checkbox" checked={!unchecked.has(adapter.style)} onChange={() => toggle(adapter.style)} className={CHECKBOX} />
          </Choice>
        ))}
      </ul>
      {chosen.length === 0 ? (
        <p className="_echo_$_mt-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
          Without a style there is nothing to rewrite with yet. You can add styles later.
        </p>
      ) : null}
      <button type="button" onClick={() => onDownload(chosen.map((adapter) => adapter.style))} className={`${PRIMARY_BUTTON} _echo_$_mt-3 _echo_$_w-full`}>
        Download · {formatBytes(total)}
      </button>
    </>
  );
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

export function ModelSetupCard({ catalog, onDownload, onRetryCatalog, notice }: ModelSetupCardProps) {
  return (
    <section
      aria-label="Writing model download"
      className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_p-4 _echo_$_shadow-card dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900"
    >
      {notice ? (
        <p role="status" className="_echo_$_mb-3 _echo_$_rounded-lg _echo_$_bg-better-200/40 _echo_$_px-2.5 _echo_$_py-1.5 _echo_$_text-xs _echo_$_font-medium _echo_$_text-ink-800 dark:_echo_$_bg-better-600/20 dark:_echo_$_text-better-200">
          {notice}
        </p>
      ) : null}

      <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-2">
        <DownloadGlyph />
        <span className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">Writing model needed</span>
      </div>
      <p className="_echo_$_mt-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{SETUP_DETAIL}</p>

      {catalog.status === 'ready' ? <Choices catalog={catalog.catalog} onDownload={onDownload} /> : null}
      {catalog.status === 'loading' ? (
        <p className="_echo_$_mt-3 _echo_$_flex _echo_$_items-center _echo_$_gap-2 _echo_$_text-xs _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
          <EchoLoader size={14} label="Checking" />
          Checking what's available…
        </p>
      ) : null}
      {catalog.status === 'failed' ? (
        <div className="_echo_$_mt-3 _echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
          <p role="alert" className="_echo_$_text-xs _echo_$_leading-5 _echo_$_text-danger-500">
            Couldn't reach the download server. Check your connection, then try again.
          </p>
          <button type="button" onClick={onRetryCatalog} className={SECONDARY_BUTTON}>
            Try again
          </button>
        </div>
      ) : null}

      <ModelSourceUrl />
    </section>
  );
}
