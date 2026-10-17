import { useId, type ReactNode } from 'react';
import { formatBytes } from '../../shared/format';
import type { StyleReadiness } from '../../shared/modelInstall';
import { styleLabel, type StyleDescriptor, type StyleId } from '../../shared/styles';

/**
 * A tooltip for the icon beside it, on hover and on keyboard focus. It sits
 * above the icon, against the right edge (the icon is at the right end of a
 * row), and is always dark, so it reads the same on the light and dark popup.
 */
function Tip({ id, title, text, children }: { id: string; title: string; text: string; children: ReactNode }) {
  return (
    <span className="_echo_$_group _echo_$_relative _echo_$_ml-auto _echo_$_inline-flex _echo_$_shrink-0">
      {children}
      <span
        id={id}
        role="tooltip"
        className="_echo_$_theme-inverse _echo_$_pointer-events-none _echo_$_invisible _echo_$_absolute _echo_$_bottom-full _echo_$_right-0 _echo_$_z-20 _echo_$_mb-1.5 _echo_$_w-[220px] _echo_$_rounded-lg _echo_$_bg-surface _echo_$_px-3 _echo_$_py-2 _echo_$_text-left _echo_$_opacity-0 _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/20 _echo_$_transition-opacity _echo_$_duration-150 group-hover:_echo_$_visible group-hover:_echo_$_opacity-100 group-focus-within:_echo_$_visible group-focus-within:_echo_$_opacity-100"
      >
        <span className="_echo_$_block _echo_$_text-xs _echo_$_font-semibold _echo_$_leading-4 _echo_$_text-fg">{title}</span>
        <span className="_echo_$_mt-0.5 _echo_$_block _echo_$_text-xs _echo_$_leading-4 _echo_$_text-fg-secondary">{text}</span>
      </span>
    </span>
  );
}

const ICON_BOX = '_echo_$_flex _echo_$_h-7 _echo_$_w-7 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg';
const FOCUS = 'focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-focus';

function DownloadIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 2.5v8M5.6 7.4 9 10.8l3.4-3.4M3 14.5h12" />
    </svg>
  );
}

function CheckBadge() {
  return (
    <span className="_echo_$_flex _echo_$_h-[22px] _echo_$_w-[22px] _echo_$_items-center _echo_$_justify-center _echo_$_rounded-full _echo_$_bg-success-subtle _echo_$_text-success-fg">
      <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
        <path d="m3.2 7.3 2.5 2.5 5.1-5.4" />
      </svg>
    </span>
  );
}

export interface StyleStatusProps {
  readonly style: StyleDescriptor;
  /** Absent while it is not known. */
  readonly readiness: StyleReadiness | undefined;
  /** The styles whose download is running now. */
  readonly downloading: readonly StyleId[];
  /** A download is running, so no other can start. */
  readonly busy: boolean;
  readonly onDownload: (adapter: StyleId) => void;
}

/**
 * Where a style stands, as one icon at the end of its row in the popup:
 *
 *   download icon   click to download it (the size is in the tooltip)
 *   spinner         downloading now (the card above shows how far)
 *   check           installed; the tooltip says what that means
 *   muted download  not yet possible: it waits for something else (the tooltip says what)
 */
export function StyleStatus({ style, readiness, downloading, busy, onDownload }: StyleStatusProps) {
  const tip = useId();
  if (!readiness) return null;
  const name = style.label;

  if (readiness.state === 'unavailable') {
    return <span className="_echo_$_ml-auto _echo_$_shrink-0 _echo_$_text-xs _echo_$_text-fg-subtle">Not available yet</span>;
  }

  if (readiness.state === 'ready') {
    return (
      <Tip id={tip} title="Installed" text="Ready to use. Stored on this device, so it works offline.">
        <span role="img" aria-label={`${name} style installed`} aria-describedby={tip} tabIndex={0} className={`${ICON_BOX} ${FOCUS}`}>
          <CheckBadge />
        </span>
      </Tip>
    );
  }

  const { adapter } = readiness;
  const adapterName = styleLabel(adapter.style);

  if (downloading.includes(adapter.style)) {
    return (
      <Tip id={tip} title="Downloading…" text="The progress is shown above. You can close this window; the download keeps going.">
        <span role="img" aria-label={`Downloading the ${adapterName} style`} aria-describedby={tip} tabIndex={0} className={`${ICON_BOX} ${FOCUS}`}>
          <span aria-hidden="true" className="_echo_$_block _echo_$_h-[18px] _echo_$_w-[18px] _echo_$_animate-spin _echo_$_rounded-full _echo_$_border-2 _echo_$_border-solid _echo_$_border-accent/25 _echo_$_border-t-accent" />
        </span>
      </Tip>
    );
  }

  if (readiness.state === 'needs-model') {
    return (
      <Tip id={tip} title="Writing model needed" text="Download the writing model above first. Every style runs on it.">
        <span role="img" aria-label={`${name} style: download the writing model first`} aria-describedby={tip} tabIndex={0} className={`${ICON_BOX} ${FOCUS} _echo_$_text-quiet`}>
          <DownloadIcon />
        </span>
      </Tip>
    );
  }

  if (adapter.style !== style.id) {
    return (
      <Tip id={tip} title={`Needs the ${adapterName} style`} text={`${name} runs on the ${adapterName} style. Download it from its own row.`}>
        <span role="img" aria-label={`${name} style needs the ${adapterName} style first`} aria-describedby={tip} tabIndex={0} className={`${ICON_BOX} ${FOCUS} _echo_$_text-quiet`}>
          <DownloadIcon />
        </span>
      </Tip>
    );
  }

  const size = formatBytes(adapter.sizeBytes);
  return (
    <Tip id={tip} title={`Download ${name}`} text={`One-time download · ${size}. Stored on this device.`}>
      <button
        type="button"
        disabled={busy}
        onClick={() => onDownload(adapter.style)}
        aria-label={`Download the ${name} style (${size})`}
        aria-describedby={tip}
        className={`${ICON_BOX} _echo_$_cursor-pointer _echo_$_border-0 _echo_$_bg-transparent _echo_$_p-0 _echo_$_text-accent-fg hover:_echo_$_bg-accent-subtle disabled:_echo_$_cursor-default disabled:_echo_$_opacity-50 disabled:hover:_echo_$_bg-transparent ${FOCUS}`}
      >
        <DownloadIcon />
      </button>
    </Tip>
  );
}
