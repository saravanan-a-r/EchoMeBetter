/**
 * First run, nothing on this device yet: one download that is enough to
 * start -- the writing model every style runs on, with the starter style
 * (Professional). Every other style is added later, from the popup or right
 * when it is first picked on a page.
 */
import { formatBytes, formatList } from '../../shared/format';
import { adapterForStyle, starterAdapter, stylesUsing, type ModelCatalog } from '../../shared/modelInstall';
import { STYLE_IDS, STYLES, styleLabel, type StyleId } from '../../shared/styles';
import { EchoLoader } from '../brand/EchoLoader';
import type { CatalogState } from '../hooks/useModel';
import { PRIMARY_BUTTON, SECONDARY_BUTTON } from './buttons';

export interface ModelSetupCardProps {
  readonly catalog: CatalogState;
  readonly onDownload: (adapters: StyleId[]) => void;
  readonly onRetryCatalog: () => void;
  /** A short confirmation shown above the card's text, e.g. right after the model was removed. */
  readonly notice?: string;
}

/** "Rewrites run entirely on your device. Download the writing model and the Professional style once; ..." */
export function setupDetail(starter: StyleId | null): string {
  const what = starter ? `the writing model and the ${styleLabel(starter)} style` : 'the writing model';
  return `Rewrites run entirely on your device. Download ${what} once. They're stored privately and used on every website.`;
}

/** "Also powers Grammar and Concise", when other styles run on this one's adapter. */
function borrowedBy(catalog: ModelCatalog, adapter: StyleId): string | null {
  const others = stylesUsing(catalog, adapter).filter((style) => style !== adapter);
  return others.length > 0 ? `Also powers ${formatList(others.map(styleLabel))}` : null;
}

/** The styles the starter download does not cover but the catalog offers: the ones to add later. */
export function laterStyles(catalog: ModelCatalog, starter: StyleId | null): StyleId[] {
  return STYLE_IDS.filter((style) => {
    const adapter = adapterForStyle(catalog, style);
    return adapter !== null && adapter !== starter && catalog.adapters.some((candidate) => candidate.style === adapter);
  });
}

function Item({ label, sizeBytes, note }: { label: string; sizeBytes: number; note: string }) {
  return (
    <li className="_echo_$_flex _echo_$_items-start _echo_$_gap-2.5 _echo_$_px-3 _echo_$_py-2">
      <span aria-hidden="true" className="_echo_$_mt-0.5 _echo_$_flex _echo_$_h-4 _echo_$_w-4 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-full _echo_$_bg-success-subtle _echo_$_text-success-fg">
        <svg width="10" height="10" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="m3.2 7.3 2.5 2.5 5.1-5.4" />
        </svg>
      </span>
      <span className="_echo_$_min-w-0 _echo_$_flex-1 _echo_$_text-sm">
        <span className="_echo_$_flex _echo_$_items-baseline _echo_$_justify-between _echo_$_gap-2">
          <span className="_echo_$_font-medium _echo_$_text-fg">{label}</span>
          <span className="_echo_$_shrink-0 _echo_$_text-xs _echo_$_tabular-nums _echo_$_text-fg-muted">{formatBytes(sizeBytes)}</span>
        </span>
        <span className="_echo_$_block _echo_$_text-xs _echo_$_text-fg-muted">{note}</span>
      </span>
    </li>
  );
}

function Starter({ catalog, onDownload }: { catalog: ModelCatalog; onDownload: (adapters: StyleId[]) => void }) {
  const starter = starterAdapter(catalog);
  const total = catalog.model.sizeBytes + (starter?.sizeBytes ?? 0);
  const later = laterStyles(catalog, starter?.style ?? null);
  const hint = starter ? STYLES.find((style) => style.id === starter.style)?.hint ?? '' : '';
  return (
    <>
      <ul aria-label="Included in this download" className="_echo_$_mt-3 _echo_$_divide-y _echo_$_divide-line _echo_$_rounded-xl _echo_$_border _echo_$_border-line">
        <Item label="Writing model" sizeBytes={catalog.model.sizeBytes} note="Powers every style" />
        {starter ? <Item label={`${styleLabel(starter.style)} style`} sizeBytes={starter.sizeBytes} note={borrowedBy(catalog, starter.style) ?? hint} /> : null}
      </ul>
      <button type="button" onClick={() => onDownload(starter ? [starter.style] : [])} className={`${PRIMARY_BUTTON} _echo_$_mt-3 _echo_$_w-full`}>
        Download · {formatBytes(total)}
      </button>
      {later.length > 0 ? (
        <p className="_echo_$_mt-2 _echo_$_text-center _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">
          Additional styles ({formatList(later.map(styleLabel))}) are optional and can be downloaded on demand after setup.
        </p>
      ) : null}
    </>
  );
}

function DownloadGlyph() {
  return (
    <span aria-hidden="true" className="_echo_$_flex _echo_$_h-5 _echo_$_w-5 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-full _echo_$_bg-accent-subtle _echo_$_text-accent-fg">
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
      className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_p-4 _echo_$_shadow-card"
    >
      {notice ? (
        <p role="status" className="_echo_$_mb-3 _echo_$_rounded-lg _echo_$_bg-success-subtle _echo_$_px-2.5 _echo_$_py-1.5 _echo_$_text-xs _echo_$_font-medium _echo_$_text-success-fg">
          {notice}
        </p>
      ) : null}

      <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-2">
        <DownloadGlyph />
        <span className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-fg">Set up EchoMeBetter</span>
      </div>
      <p className="_echo_$_mt-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">
        {setupDetail(catalog.status === 'ready' ? starterAdapter(catalog.catalog)?.style ?? null : 'professional')}
      </p>

      {catalog.status === 'ready' ? <Starter catalog={catalog.catalog} onDownload={onDownload} /> : null}
      {catalog.status === 'loading' ? (
        <p className="_echo_$_mt-3 _echo_$_flex _echo_$_items-center _echo_$_gap-2 _echo_$_text-xs _echo_$_text-fg-muted">
          <EchoLoader size={14} label="Checking" />
          Checking what's available…
        </p>
      ) : null}
      {catalog.status === 'failed' ? (
        <div className="_echo_$_mt-3 _echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
          <p role="alert" className="_echo_$_text-xs _echo_$_leading-5 _echo_$_text-danger">
            Couldn't reach the download server. Check your connection, then try again.
          </p>
          <button type="button" onClick={onRetryCatalog} className={SECONDARY_BUTTON}>
            Try again
          </button>
        </div>
      ) : null}
    </section>
  );
}
