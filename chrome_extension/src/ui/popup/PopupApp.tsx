import { useState } from 'react';
import type { UiReply } from '../../shared/messages';
import type { KeyPlatform } from '../../shared/shortcuts';
import type { EngineStatus } from '../../shared/status';
import { Logo, Wordmark } from '../brand/Logo';
import { ModelDownloadCard } from '../components/ModelDownloadCard';
import { RemoveModel } from '../components/RemoveModel';
import { StatusCard } from '../components/StatusCard';
import { ShortcutAccessCard } from '../components/ShortcutAccessCard';
import { StyleList } from '../components/StyleList';
import { requestWarmUp, useEngineStatus } from '../hooks/useEngineStatus';
import { requestCancelDownload, requestDownload, requestRemoveModel, useModelAvailability, type ModelAvailability } from '../hooks/useModel';
import { useSettings } from '../hooks/useSettings';
import { PLATFORM, releaseSiteAccess, requestSiteAccess, shortcutAvailability, useSiteAccess, type ShortcutAvailability } from '../hooks/useShortcuts';
import { SettingsView } from './SettingsView';

const STEPS = ['Right-click', 'EchoMeBetter'];

export const REMOVED_NOTICE = 'Model removed. Download it again whenever you want to rewrite text.';

export interface PopupViewProps {
  readonly status: EngineStatus;
  readonly model: ModelAvailability;
  /** The model was removed from this popup; say so until a new download starts. */
  readonly modelRemoved: boolean;
  readonly platform: KeyPlatform;
  readonly shortcuts: ShortcutAvailability;
  readonly onAllowSiteAccess: () => void;
  readonly onTurnOffShortcuts: () => void;
  readonly onOpenSettings: () => void;
  readonly onLoadModel: () => void;
  readonly onDownloadModel: () => void;
  readonly onCancelDownload: () => void;
  readonly onRemoveModel: () => Promise<UiReply>;
  readonly onOpenGuide: () => void;
}

export function PopupView(props: PopupViewProps) {
  const { status, model, modelRemoved, platform, shortcuts, onLoadModel, onOpenGuide } = props;
  return (
    <main className="_echo_$_w-[360px] _echo_$_bg-ink-50 _echo_$_text-ink-900 dark:_echo_$_bg-ink-950 dark:_echo_$_text-white">
      <header className="_echo_$_relative _echo_$_overflow-hidden _echo_$_bg-echo-gradient _echo_$_px-4 _echo_$_pb-5 _echo_$_pt-3.5 _echo_$_text-white">
        <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-2.5">
          <span className="_echo_$_rounded-xl _echo_$_bg-white/15 _echo_$_p-0.5 _echo_$_ring-1 _echo_$_ring-white/30">
            <Logo size={30} />
          </span>
          <div>
            <Wordmark className="_echo_$_text-base [&>span]:_echo_$_text-better-200" />
            <p className="_echo_$_text-xs _echo_$_text-white/80">Rewrite anything, right where you type.</p>
          </div>
          <button
            type="button"
            aria-label="Settings"
            title="Settings"
            onClick={props.onOpenSettings}
            className="_echo_$_ml-auto _echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_items-center _echo_$_justify-center _echo_$_self-start _echo_$_rounded-lg _echo_$_text-white/90 hover:_echo_$_bg-white/15 hover:_echo_$_text-white focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-white"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
            </svg>
          </button>
        </div>
      </header>

      <div className="_echo_$_space-y-3.5 _echo_$_px-4 _echo_$_pb-3 _echo_$_pt-0">
        <div className="_echo_$_mt-[-12px] _echo_$_relative">
          {!model.known ? null : model.installed ? (
            <StatusCard status={status} onLoad={onLoadModel} />
          ) : (
            <ModelDownloadCard
              download={model.download}
              onDownload={props.onDownloadModel}
              onCancel={props.onCancelDownload}
              notice={modelRemoved ? REMOVED_NOTICE : undefined}
            />
          )}
        </div>

        <p className="_echo_$_flex _echo_$_flex-wrap _echo_$_items-center _echo_$_gap-1.5 _echo_$_text-[13px] _echo_$_text-ink-600 dark:_echo_$_text-ink-200">
          Select text
          {STEPS.map((step) => (
            <span key={step} className="_echo_$_flex _echo_$_items-center _echo_$_gap-1.5">
              <span aria-hidden="true" className="_echo_$_text-ink-300">›</span>
              <kbd className="_echo_$_rounded-md _echo_$_border _echo_$_border-ink-200 _echo_$_bg-white _echo_$_px-1.5 _echo_$_py-0.5 _echo_$_font-sans _echo_$_text-xs _echo_$_font-semibold _echo_$_text-ink-800 dark:_echo_$_border-ink-700 dark:_echo_$_bg-ink-900 dark:_echo_$_text-ink-100">
                {step}
              </kbd>
            </span>
          ))}
          <span aria-hidden="true" className="_echo_$_text-ink-300">›</span>
          pick a style
        </p>

        <section aria-labelledby="styles-title">
          <h2 id="styles-title" className="_echo_$_mb-2 _echo_$_text-[11px] _echo_$_font-semibold _echo_$_uppercase _echo_$_tracking-wider _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
            Styles
          </h2>
          <StyleList dense shortcuts={shortcuts === 'on' ? platform : undefined} />
        </section>

        {shortcuts === 'needs-access' ? (
          <ShortcutAccessCard platform={platform} onAllow={props.onAllowSiteAccess} onNotNow={props.onTurnOffShortcuts} />
        ) : null}

        {model.installed ? <RemoveModel sizeBytes={model.installed.model.sizeBytes} onRemove={props.onRemoveModel} /> : null}

        <footer className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_text-xs _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
          <span className="_echo_$_flex _echo_$_items-center _echo_$_gap-1.5">
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
              <path d="M6 1 2 2.6v3c0 2.4 1.7 4.4 4 5.2 2.3-.8 4-2.8 4-5.2v-3L6 1Z" fill="none" stroke="currentColor" strokeWidth="1.2" />
            </svg>
            On-device. Your text never leaves the browser.
          </span>
          <button type="button" onClick={onOpenGuide} className="_echo_$_font-semibold _echo_$_text-echo-600 hover:_echo_$_underline dark:_echo_$_text-echo-300">
            Guide
          </button>
        </footer>
      </div>
    </main>
  );
}

export function PopupApp() {
  const status = useEngineStatus();
  const model = useModelAvailability();
  const [settings, setSettings, settingsLoaded] = useSettings();
  const siteAccess = useSiteAccess();
  const [modelRemoved, setModelRemoved] = useState(false);
  const [page, setPage] = useState<'home' | 'settings'>('home');
  const shortcuts = shortcutAvailability(settingsLoaded ? settings.shortcutsEnabled : null, siteAccess);

  const setShortcuts = (on: boolean) => {
    // Chrome only shows the permission prompt straight from a click, so ask before anything awaits.
    if (on) requestSiteAccess();
    else releaseSiteAccess();
    setSettings({ ...settings, shortcutsEnabled: on });
  };

  if (page === 'settings') {
    return (
      <SettingsView
        platform={PLATFORM}
        shortcuts={shortcuts}
        onShortcutsChange={setShortcuts}
        onAllowSiteAccess={requestSiteAccess}
        keepLoaded={settings.keepModelLoadedMinutes}
        onKeepLoadedChange={(minutes) => setSettings({ ...settings, keepModelLoadedMinutes: minutes })}
        onBack={() => setPage('home')}
      />
    );
  }

  return (
    <PopupView
      status={status}
      model={model}
      modelRemoved={modelRemoved}
      platform={PLATFORM}
      shortcuts={shortcuts}
      onAllowSiteAccess={requestSiteAccess}
      onTurnOffShortcuts={() => setShortcuts(false)}
      onOpenSettings={() => setPage('settings')}
      onLoadModel={requestWarmUp}
      onDownloadModel={() => {
        setModelRemoved(false);
        requestDownload();
      }}
      onCancelDownload={requestCancelDownload}
      onRemoveModel={async () => {
        const reply = await requestRemoveModel();
        if (reply.ok) setModelRemoved(true);
        return reply;
      }}
      onOpenGuide={() => void chrome.tabs.create({ url: chrome.runtime.getURL('ui/welcome/welcome.html') })}
    />
  );
}
