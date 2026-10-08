/**
 * The popup's settings page, opened from the gear. Each setting is a card in
 * a titled section, so new settings slot in without reshaping the page.
 *
 * Opened from a button elsewhere (a toast on a web page), it can start at
 * one setting: scrolled to, focused and highlighted, with a line on why it
 * is worth a look.
 */
import { useEffect, useRef, type ReactNode } from 'react';
import type { SettingsFocus } from '../../shared/popupIntent';
import { KEEP_LOADED_CHOICES, keepLoadedLabel, type KeepLoadedMinutes } from '../../shared/settings';
import type { KeyPlatform } from '../../shared/shortcuts';
import { STYLES } from '../../shared/styles';
import { PerformanceSettings, type PerformanceSettingsProps } from '../components/PerformanceSettings';
import { ShortcutAccessCard } from '../components/ShortcutAccessCard';
import { ShortcutKeys } from '../components/ShortcutKeys';
import { StorageSettings, type StorageSettingsProps } from '../components/StorageSettings';
import { Switch } from '../components/Switch';
import type { WebsiteAvailability } from '../hooks/useShortcuts';

export const SHORTCUT_NOTE =
  'Shortcuts only act when text is selected in a text box; every other key press goes to the website as usual. If one clashes with a site you use, turn shortcuts off here.';

export interface SettingsViewProps extends StorageSettingsProps {
  readonly platform: KeyPlatform;
  readonly shortcuts: WebsiteAvailability;
  readonly onShortcutsChange: (on: boolean) => void;
  readonly holdMenu: WebsiteAvailability;
  readonly onHoldMenuChange: (on: boolean) => void;
  readonly onAllowSiteAccess: () => void;
  readonly keepLoaded: KeepLoadedMinutes;
  readonly onKeepLoadedChange: (minutes: KeepLoadedMinutes) => void;
  /** Null until the settings have been read. */
  readonly performance?: PerformanceSettingsProps | null;
  /** The setting to bring forward, when the page was opened for it. */
  readonly focus?: SettingsFocus | null;
  readonly onBack: () => void;
}

export const HOLD_MENU_DESCRIPTION = 'Select text, then press and hold on it for 2 seconds to pick a style. No right-click needed.';

export const KEEP_AWAKE_NOTE =
  'Rewrites wait while EchoMeBetter wakes up. Pick a longer time to keep it awake between rewrites, so they start right away.';

function Section({ id, title, highlight = false, children }: { id: string; title: string; highlight?: boolean; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className={highlight ? '_echo_$_scroll-my-4' : undefined}>
      <h2 id={id} className="_echo_$_mb-2 _echo_$_text-[11px] _echo_$_font-semibold _echo_$_uppercase _echo_$_tracking-wider _echo_$_text-fg-muted">
        {title}
      </h2>
      <div className={`_echo_$_rounded-xl _echo_$_border _echo_$_bg-surface _echo_$_p-3 ${highlight ? '_echo_$_border-accent-line _echo_$_ring-4 _echo_$_ring-accent-subtle' : '_echo_$_border-line'}`}>
        {children}
      </div>
    </section>
  );
}

export function SettingsView(props: SettingsViewProps) {
  const { platform, shortcuts, holdMenu, keepLoaded } = props;
  const waiting = { shortcuts: shortcuts === 'needs-access', holdMenu: holdMenu === 'needs-access' };
  // One request for the permission, under the first feature waiting for it.
  const accessCard = (
    <div className="_echo_$_mt-3">
      <ShortcutAccessCard platform={platform} waiting={waiting} onAllow={props.onAllowSiteAccess} title="One more step: allow on websites" />
    </div>
  );
  const keepAwake = props.focus === 'keep-awake';
  const keepLoadedSelect = useRef<HTMLSelectElement>(null);
  useEffect(() => {
    if (!keepAwake) return;
    const select = keepLoadedSelect.current;
    select?.scrollIntoView({ block: 'center' });
    select?.focus({ preventScroll: true });
  }, [keepAwake]);
  return (
    <main className="_echo_$_w-[360px] _echo_$_bg-canvas _echo_$_text-fg">
      <header className="_echo_$_flex _echo_$_items-center _echo_$_gap-2 _echo_$_border-b _echo_$_border-line _echo_$_bg-surface _echo_$_px-2 _echo_$_py-2.5">
        <button
          type="button"
          aria-label="Back"
          onClick={props.onBack}
          className="_echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg _echo_$_text-fg-soft hover:_echo_$_bg-muted focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-focus"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <h1 className="_echo_$_text-base _echo_$_font-semibold">Settings</h1>
      </header>

      <div className="_echo_$_space-y-5 _echo_$_px-4 _echo_$_py-4">
        {props.performance && props.performance.gpu.state !== 'checking' ? (
          <Section id="performance-title" title="Performance">
            <PerformanceSettings {...props.performance} />
          </Section>
        ) : null}

        <Section id="shortcuts-title" title="Keyboard">
          <div className="_echo_$_flex _echo_$_items-start _echo_$_justify-between _echo_$_gap-3">
            <div>
              <p id="shortcuts-label" className="_echo_$_text-sm _echo_$_font-medium">
                Keyboard shortcuts
              </p>
              <p id="shortcuts-description" className="_echo_$_mt-0.5 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">
                Select text, then press a shortcut to rewrite it without the right-click menu.
              </p>
            </div>
            {shortcuts === 'loading' ? null : (
              <Switch checked={shortcuts !== 'off'} onChange={props.onShortcutsChange} labelledBy="shortcuts-label" describedBy="shortcuts-description" />
            )}
          </div>

          {waiting.shortcuts ? accessCard : null}

          {shortcuts === 'on' || shortcuts === 'needs-access' ? (
            <ul aria-label="Shortcuts" className={`_echo_$_mt-3 _echo_$_space-y-1.5 ${shortcuts === 'needs-access' ? '_echo_$_opacity-60' : ''}`}>
              {STYLES.map((style) => (
                <li key={style.id} className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_text-sm">
                  <span>{style.label}</span>
                  <ShortcutKeys style={style.id} platform={platform} />
                </li>
              ))}
            </ul>
          ) : null}

          <p className="_echo_$_mt-3 _echo_$_border-t _echo_$_border-line _echo_$_pt-2.5 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">
            {SHORTCUT_NOTE}
          </p>
        </Section>

        <Section id="hold-menu-title" title="Mouse">
          <div className="_echo_$_flex _echo_$_items-start _echo_$_justify-between _echo_$_gap-3">
            <div>
              <p id="hold-menu-label" className="_echo_$_text-sm _echo_$_font-medium">
                Press and hold menu
              </p>
              <p id="hold-menu-description" className="_echo_$_mt-0.5 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted">
                {HOLD_MENU_DESCRIPTION}
              </p>
            </div>
            {holdMenu === 'loading' ? null : (
              <Switch checked={holdMenu !== 'off'} onChange={props.onHoldMenuChange} labelledBy="hold-menu-label" describedBy="hold-menu-description" />
            )}
          </div>
          {waiting.holdMenu && !waiting.shortcuts ? accessCard : null}
        </Section>

        <Section id="memory-title" title="Memory" highlight={keepAwake}>
          {keepAwake ? (
            <p id="keep-awake-note" className="_echo_$_mb-3 _echo_$_rounded-lg _echo_$_bg-accent-subtle _echo_$_px-2.5 _echo_$_py-2 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-accent-fg-strong">
              {KEEP_AWAKE_NOTE}
            </p>
          ) : null}
          <div className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
            <label htmlFor="keep-loaded" className="_echo_$_text-sm">
              <span className="_echo_$_block _echo_$_font-medium">Free memory when idle for</span>
              <span className="_echo_$_block _echo_$_text-xs _echo_$_text-fg-muted">Uses about 1 GB while loaded</span>
            </label>
            <select
              id="keep-loaded"
              ref={keepLoadedSelect}
              aria-describedby={keepAwake ? 'keep-awake-note' : undefined}
              value={keepLoaded}
              onChange={(event) => props.onKeepLoadedChange(Number(event.target.value) as KeepLoadedMinutes)}
              className="_echo_$_rounded-lg _echo_$_border _echo_$_border-line-strong _echo_$_bg-surface-raised _echo_$_px-2 _echo_$_py-1 _echo_$_text-sm"
            >
              {KEEP_LOADED_CHOICES.map((minutes) => (
                <option key={minutes} value={minutes}>
                  {keepLoadedLabel(minutes)}
                </option>
              ))}
            </select>
          </div>
        </Section>

        <Section id="storage-title" title="Storage">
          <StorageSettings installed={props.installed} onRemoveModel={props.onRemoveModel} onRemoveAdapter={props.onRemoveAdapter} />
        </Section>
      </div>
    </main>
  );
}
