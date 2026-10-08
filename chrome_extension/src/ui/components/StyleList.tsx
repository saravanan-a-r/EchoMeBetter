import type { ReactNode } from 'react';
import { formatBytes } from '../../shared/format';
import type { StyleReadiness } from '../../shared/modelInstall';
import type { KeyPlatform } from '../../shared/shortcuts';
import { STYLES, styleLabel, type StyleDescriptor, type StyleId } from '../../shared/styles';
import { StyleGlyph } from '../brand/StyleGlyph';
import { SECONDARY_BUTTON } from './buttons';
import { ShortcutKeys } from './ShortcutKeys';

export interface StyleListProps {
  readonly columns?: 1 | 2 | 3;
  /** One bordered card with a row per style, sized for the 600px-high toolbar popup. */
  readonly dense?: boolean;
  /** Show each style's shortcut, written for that platform (only while shortcuts work). */
  readonly shortcuts?: KeyPlatform;
  /** Per style: ready, or what it still needs. Absent while that is not known. */
  readonly readiness?: Readonly<Partial<Record<StyleId, StyleReadiness>>>;
  readonly onDownload?: (adapter: StyleId) => void;
  /** A download is running, so no other can start. */
  readonly busy?: boolean;
}

/** The adapter the style runs with, when that is known. */
function runsWith(readiness: StyleReadiness | undefined): StyleId | null {
  if (!readiness || readiness.state === 'unavailable') return null;
  return readiness.state === 'ready' ? readiness.adapter : readiness.adapter.style;
}

/** A style still in training runs with another style's adapter for now; say so. */
function PreviewBadge({ style, adapter }: { style: StyleId; adapter: StyleId }) {
  const explanation = `${styleLabel(style)}'s own style is still in training. Until it's ready, ${styleLabel(style)} rewrites with the ${styleLabel(adapter)} style.`;
  return (
    <span
      title={explanation}
      className="_echo_$_shrink-0 _echo_$_rounded-full _echo_$_bg-accent-subtle _echo_$_px-1.5 _echo_$_py-0.5 _echo_$_text-[10px] _echo_$_font-semibold _echo_$_uppercase _echo_$_tracking-wide _echo_$_text-accent-fg-strong"
    >
      Preview<span className="_echo_$_sr-only">: {explanation}</span>
    </span>
  );
}

/** What stands to the right of a style: a way to get it, what it waits for, or its shortcut. */
function StyleEnd({ style, props, fallback }: { style: StyleDescriptor; props: StyleListProps; fallback: ReactNode }) {
  const readiness = props.readiness?.[style.id];
  if (readiness?.state === 'unavailable') return <span className="_echo_$_ml-auto _echo_$_shrink-0 _echo_$_text-xs _echo_$_text-fg-subtle">Not available yet</span>;
  if (readiness?.state !== 'needs-download') return <>{fallback}</>;
  const { adapter } = readiness;
  if (adapter.style !== style.id) {
    return <span className="_echo_$_ml-auto _echo_$_shrink-0 _echo_$_text-xs _echo_$_text-fg-muted">Needs {styleLabel(adapter.style)}</span>;
  }
  return (
    <button
      type="button"
      disabled={props.busy}
      onClick={() => props.onDownload?.(adapter.style)}
      aria-label={`Download the ${style.label} style (${formatBytes(adapter.sizeBytes)})`}
      className={`${SECONDARY_BUTTON} _echo_$_ml-auto`}
    >
      Get · {formatBytes(adapter.sizeBytes)}
    </button>
  );
}

function Label({ style, readiness, className }: { style: StyleDescriptor; readiness: StyleReadiness | undefined; className: string }) {
  const adapter = runsWith(readiness);
  return (
    <span className="_echo_$_flex _echo_$_min-w-0 _echo_$_items-center _echo_$_gap-1.5">
      <span className={className}>{style.label}</span>
      {adapter && adapter !== style.id ? <PreviewBadge style={style.id} adapter={adapter} /> : null}
    </span>
  );
}

export function StyleList(props: StyleListProps) {
  const { columns = 1, dense = false, shortcuts, readiness } = props;
  if (dense) {
    return (
      <ul className="_echo_$_divide-y _echo_$_divide-line _echo_$_overflow-hidden _echo_$_rounded-xl _echo_$_border _echo_$_border-line _echo_$_bg-surface">
        {STYLES.map((style) => (
          <li key={style.id} className="_echo_$_flex _echo_$_items-center _echo_$_gap-2.5 _echo_$_px-3 _echo_$_py-2">
            <span className="_echo_$_flex _echo_$_h-6 _echo_$_w-6 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-md _echo_$_bg-accent-subtle _echo_$_text-accent-fg">
              <StyleGlyph style={style.id} size={14} />
            </span>
            <Label style={style} readiness={readiness?.[style.id]} className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-fg" />
            <StyleEnd
              style={style}
              props={props}
              fallback={
                // A row has room for the hint or the shortcut; the shortcut is the one to learn here.
                shortcuts ? (
                  <span title={style.hint} className="_echo_$_ml-auto">
                    <ShortcutKeys style={style.id} platform={shortcuts} compact />
                  </span>
                ) : (
                  <span className="_echo_$_ml-auto _echo_$_truncate _echo_$_text-xs _echo_$_text-fg-muted">{style.hint}</span>
                )
              }
            />
          </li>
        ))}
      </ul>
    );
  }
  const grid = columns === 1 ? '' : columns === 2 ? 'sm:_echo_$_grid-cols-2' : 'sm:_echo_$_grid-cols-2 lg:_echo_$_grid-cols-3';
  return (
    <ul className={`_echo_$_grid _echo_$_gap-2 ${grid}`}>
      {STYLES.map((style) => (
        <li
          key={style.id}
          className="_echo_$_flex _echo_$_items-center _echo_$_gap-3 _echo_$_rounded-xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_px-3 _echo_$_py-2.5"
        >
          <span className="_echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg _echo_$_bg-accent-subtle _echo_$_text-accent-fg">
            <StyleGlyph style={style.id} />
          </span>
          <span className="_echo_$_min-w-0 _echo_$_flex-1">
            <Label style={style} readiness={readiness?.[style.id]} className="_echo_$_block _echo_$_text-sm _echo_$_font-semibold _echo_$_text-fg" />
            <span className="_echo_$_block _echo_$_text-xs _echo_$_text-fg-muted">{style.hint}</span>
          </span>
          <StyleEnd style={style} props={props} fallback={shortcuts ? <ShortcutKeys style={style.id} platform={shortcuts} /> : null} />
        </li>
      ))}
    </ul>
  );
}
