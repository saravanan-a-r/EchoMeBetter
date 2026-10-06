/**
 * The popup's settings page, opened from the gear. Each setting is a card in
 * a titled section, so new settings slot in without reshaping the page.
 */
import type { ReactNode } from 'react';
import { KEEP_LOADED_CHOICES, type KeepLoadedMinutes } from '../../shared/settings';
import type { KeyPlatform } from '../../shared/shortcuts';
import { STYLES } from '../../shared/styles';
import { ShortcutAccessCard } from '../components/ShortcutAccessCard';
import { ShortcutKeys } from '../components/ShortcutKeys';
import { StorageSettings, type StorageSettingsProps } from '../components/StorageSettings';
import { Switch } from '../components/Switch';
import type { ShortcutAvailability } from '../hooks/useShortcuts';

const KEEP_LABELS: Record<KeepLoadedMinutes, string> = { 5: '5 minutes', 15: '15 minutes', 60: '1 hour', 0: 'Never' };

export const SHORTCUT_NOTE =
  'Shortcuts only act when text is selected in a text box; every other key press goes to the website as usual. If one clashes with a site you use, turn shortcuts off here.';

export interface SettingsViewProps extends StorageSettingsProps {
  readonly platform: KeyPlatform;
  readonly shortcuts: ShortcutAvailability;
  readonly onShortcutsChange: (on: boolean) => void;
  readonly onAllowSiteAccess: () => void;
  readonly keepLoaded: KeepLoadedMinutes;
  readonly onKeepLoadedChange: (minutes: KeepLoadedMinutes) => void;
  readonly onBack: () => void;
}

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id}>
      <h2 id={id} className="_echo_$_mb-2 _echo_$_text-[11px] _echo_$_font-semibold _echo_$_uppercase _echo_$_tracking-wider _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
        {title}
      </h2>
      <div className="_echo_$_rounded-xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_p-3 dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900">{children}</div>
    </section>
  );
}

export function SettingsView(props: SettingsViewProps) {
  const { platform, shortcuts, keepLoaded } = props;
  return (
    <main className="_echo_$_w-[360px] _echo_$_bg-ink-50 _echo_$_text-ink-900 dark:_echo_$_bg-ink-950 dark:_echo_$_text-white">
      <header className="_echo_$_flex _echo_$_items-center _echo_$_gap-2 _echo_$_border-b _echo_$_border-ink-100 _echo_$_bg-white _echo_$_px-2 _echo_$_py-2.5 dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900">
        <button
          type="button"
          aria-label="Back"
          onClick={props.onBack}
          className="_echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg _echo_$_text-ink-600 hover:_echo_$_bg-ink-100 focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-echo-500 dark:_echo_$_text-ink-200 dark:hover:_echo_$_bg-ink-800"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M10 3 5 8l5 5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        <h1 className="_echo_$_text-base _echo_$_font-semibold">Settings</h1>
      </header>

      <div className="_echo_$_space-y-5 _echo_$_px-4 _echo_$_py-4">
        <Section id="shortcuts-title" title="Keyboard">
          <div className="_echo_$_flex _echo_$_items-start _echo_$_justify-between _echo_$_gap-3">
            <div>
              <p id="shortcuts-label" className="_echo_$_text-sm _echo_$_font-medium">
                Keyboard shortcuts
              </p>
              <p id="shortcuts-description" className="_echo_$_mt-0.5 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
                Select text, then press a shortcut to rewrite it without the right-click menu.
              </p>
            </div>
            {shortcuts === 'loading' ? null : (
              <Switch checked={shortcuts !== 'off'} onChange={props.onShortcutsChange} labelledBy="shortcuts-label" describedBy="shortcuts-description" />
            )}
          </div>

          {shortcuts === 'needs-access' ? (
            <div className="_echo_$_mt-3">
              <ShortcutAccessCard platform={platform} onAllow={props.onAllowSiteAccess} title="One more step: allow on websites" />
            </div>
          ) : null}

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

          <p className="_echo_$_mt-3 _echo_$_border-t _echo_$_border-ink-100 _echo_$_pt-2.5 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-500 dark:_echo_$_border-ink-800 dark:_echo_$_text-ink-300">
            {SHORTCUT_NOTE}
          </p>
        </Section>

        <Section id="memory-title" title="Memory">
          <div className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
            <label htmlFor="keep-loaded" className="_echo_$_text-sm">
              <span className="_echo_$_block _echo_$_font-medium">Free memory when idle for</span>
              <span className="_echo_$_block _echo_$_text-xs _echo_$_text-ink-500 dark:_echo_$_text-ink-300">Uses about 1 GB while loaded</span>
            </label>
            <select
              id="keep-loaded"
              value={keepLoaded}
              onChange={(event) => props.onKeepLoadedChange(Number(event.target.value) as KeepLoadedMinutes)}
              className="_echo_$_rounded-lg _echo_$_border _echo_$_border-ink-200 _echo_$_bg-white _echo_$_px-2 _echo_$_py-1 _echo_$_text-sm dark:_echo_$_border-ink-700 dark:_echo_$_bg-ink-800"
            >
              {KEEP_LOADED_CHOICES.map((minutes) => (
                <option key={minutes} value={minutes}>
                  {KEEP_LABELS[minutes]}
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
