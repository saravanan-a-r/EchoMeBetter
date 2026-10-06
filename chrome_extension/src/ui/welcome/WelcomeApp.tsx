import { shortcutLabel, type KeyPlatform } from '../../shared/shortcuts';
import type { EngineStatus } from '../../shared/status';
import { Logo, Wordmark } from '../brand/Logo';
import { ContextMenuPreview } from '../components/ContextMenuPreview';
import { ModelDownloadCard } from '../components/ModelDownloadCard';
import { ShortcutAccessCard } from '../components/ShortcutAccessCard';
import { StatusCard } from '../components/StatusCard';
import { StyleList } from '../components/StyleList';
import { requestWarmUp, useEngineStatus } from '../hooks/useEngineStatus';
import { requestCancelDownload, requestDownload, useModelAvailability, type ModelAvailability } from '../hooks/useModel';
import { useSettings } from '../hooks/useSettings';
import { PLATFORM, requestSiteAccess, shortcutAvailability, useSiteAccess, type ShortcutAvailability } from '../hooks/useShortcuts';

export const PLAYGROUND_SAMPLE =
  "hey team, just wanted to let u know the release is gonna slip to next week because QA found a couple of bugs in the checkout flow. sorry about that, we're on it";

const JOURNEY = [
  { title: 'Select', body: 'Highlight a sentence or paragraph in any text box: email, chat, docs, forms.' },
  { title: 'Right-click', body: 'Choose EchoMeBetter and hover to see the styles.' },
  { title: 'Pick a style', body: 'Your pointer shows the rewrite is in progress. Press Esc to cancel.' },
  { title: 'Keep or undo', body: 'The rewrite replaces your selection. One click on Undo brings the original back.' },
];

export interface WelcomeViewProps {
  readonly status: EngineStatus;
  readonly model: ModelAvailability;
  readonly onLoadModel: () => void;
  readonly onDownloadModel: () => void;
  readonly onCancelDownload: () => void;
  readonly platform: KeyPlatform;
  readonly shortcuts: ShortcutAvailability;
  readonly onAllowSiteAccess: () => void;
}

export function WelcomeView({ status, model, onLoadModel, onDownloadModel, onCancelDownload, platform, shortcuts, onAllowSiteAccess }: WelcomeViewProps) {
  const needsDownload = model.known && !model.installed;
  // This page listens for shortcuts itself, so they work here even before websites are allowed.
  const showKeys = shortcuts === 'on' || shortcuts === 'needs-access' ? platform : undefined;
  return (
    <div className="_echo_$_min-h-screen _echo_$_bg-ink-50 _echo_$_text-ink-900 dark:_echo_$_bg-ink-950 dark:_echo_$_text-white">
      <header className="_echo_$_relative _echo_$_overflow-hidden _echo_$_bg-ink-950 _echo_$_text-white">
        <div aria-hidden="true" className="_echo_$_absolute _echo_$_inset-0 _echo_$_bg-echo-glow" />
        <div className="_echo_$_relative _echo_$_mx-auto _echo_$_flex _echo_$_max-w-5xl _echo_$_flex-col _echo_$_items-center _echo_$_px-6 _echo_$_pb-16 _echo_$_pt-14 _echo_$_text-center">
          <Logo size={72} />
          <h1 className="_echo_$_mt-6 _echo_$_text-4xl _echo_$_font-bold _echo_$_tracking-tight sm:_echo_$_text-5xl">
            Say it better, <span className="_echo_$_bg-echo-gradient _echo_$_bg-clip-text _echo_$_text-transparent">right where you type.</span>
          </h1>
          <p className="_echo_$_mt-4 _echo_$_max-w-2xl _echo_$_text-lg _echo_$_text-ink-200">
            <Wordmark /> rewrites your selected text, whether you want it professional, corrected, friendly, concise or elaborate, with a
            writing model that runs entirely on your device.
          </p>
        </div>
      </header>

      <main className="_echo_$_mx-auto _echo_$_max-w-5xl _echo_$_space-y-14 _echo_$_px-6 _echo_$_py-12">
        <section aria-labelledby="journey-title" className="_echo_$_grid _echo_$_items-center _echo_$_gap-10 md:_echo_$_grid-cols-[1fr_auto]">
          <div>
            <h2 id="journey-title" className="_echo_$_text-2xl _echo_$_font-bold">Four steps, no copy and paste</h2>
            <ol className="_echo_$_mt-6 _echo_$_space-y-5">
              {JOURNEY.map((step, index) => (
                <li key={step.title} className="_echo_$_flex _echo_$_gap-4">
                  <span className="_echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-full _echo_$_bg-echo-600 _echo_$_text-sm _echo_$_font-bold _echo_$_text-white">
                    {index + 1}
                  </span>
                  <div>
                    <h3 className="_echo_$_font-semibold">{step.title}</h3>
                    <p className="_echo_$_text-sm _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{step.body}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
          <ContextMenuPreview shortcuts={showKeys} />
        </section>

        <section aria-labelledby="try-title" className="_echo_$_rounded-3xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_p-6 _echo_$_shadow-card dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900">
          <div className="_echo_$_flex _echo_$_flex-wrap _echo_$_items-start _echo_$_justify-between _echo_$_gap-4">
            <div>
              <h2 id="try-title" className="_echo_$_text-2xl _echo_$_font-bold">Try it here</h2>
              <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
                {needsDownload
                  ? 'Download the model, then select text below, right-click and choose EchoMeBetter.'
                  : 'Select some of the text below, right-click, and choose EchoMeBetter.'}
              </p>
              {showKeys ? (
                <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_text-ink-500 dark:_echo_$_text-ink-300">
                  Or use the keyboard: select text and press {shortcutLabel('professional', showKeys)} for Professional.
                </p>
              ) : null}
            </div>
            <div className="_echo_$_w-full sm:_echo_$_w-80">
              {!model.known ? null : model.installed ? (
                <StatusCard status={status} onLoad={onLoadModel} />
              ) : (
                <ModelDownloadCard download={model.download} onDownload={onDownloadModel} onCancel={onCancelDownload} />
              )}
            </div>
          </div>
          {shortcuts === 'needs-access' ? (
            <div className="_echo_$_mt-5">
              <ShortcutAccessCard platform={platform} onAllow={onAllowSiteAccess} />
            </div>
          ) : null}
          <label htmlFor="playground" className="_echo_$_sr-only">
            Practice text
          </label>
          <textarea
            id="playground"
            defaultValue={PLAYGROUND_SAMPLE}
            rows={5}
            spellCheck={false}
            className="_echo_$_mt-5 _echo_$_w-full _echo_$_resize-y _echo_$_rounded-2xl _echo_$_border _echo_$_border-ink-200 _echo_$_bg-ink-50 _echo_$_p-4 _echo_$_text-base _echo_$_leading-7 _echo_$_text-ink-900 _echo_$_outline-none focus:_echo_$_border-echo-400 focus:_echo_$_ring-4 focus:_echo_$_ring-echo-100 dark:_echo_$_border-ink-700 dark:_echo_$_bg-ink-950 dark:_echo_$_text-white dark:focus:_echo_$_ring-echo-950"
          />
        </section>

        <section aria-labelledby="styles-title">
          <h2 id="styles-title" className="_echo_$_text-2xl _echo_$_font-bold">Five styles, one right-click away</h2>
          <div className="_echo_$_mt-6">
            <StyleList columns={3} shortcuts={showKeys} />
          </div>
        </section>

        <section className="_echo_$_grid _echo_$_gap-4 md:_echo_$_grid-cols-3">
          {[
            ['Private by design', 'The model runs inside your browser. Your text is never uploaded, logged or shared.'],
            ['Light on your browser', 'Rewrites run in a background worker, so the page you are typing in stays responsive. The model unloads itself when idle.'],
            ['Works where you write', 'Text areas, input fields and rich editors such as email composers. Native undo keeps working.'],
          ].map(([title, body]) => (
            <div key={title} className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-ink-100 _echo_$_bg-white _echo_$_p-5 dark:_echo_$_border-ink-800 dark:_echo_$_bg-ink-900">
              <h3 className="_echo_$_font-semibold">{title}</h3>
              <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_leading-6 _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{body}</p>
            </div>
          ))}
        </section>
      </main>

      <footer className="_echo_$_border-t _echo_$_border-ink-100 _echo_$_py-6 _echo_$_text-center _echo_$_text-xs _echo_$_text-ink-500 dark:_echo_$_border-ink-800 dark:_echo_$_text-ink-300">
        EchoMeBetter · Pin it from the puzzle-piece menu to see the model status at a glance.
      </footer>
    </div>
  );
}

export function WelcomeApp() {
  const status = useEngineStatus();
  const model = useModelAvailability();
  const [settings, , settingsLoaded] = useSettings();
  const siteAccess = useSiteAccess();
  return (
    <WelcomeView
      platform={PLATFORM}
      shortcuts={shortcutAvailability(settingsLoaded ? settings.shortcutsEnabled : null, siteAccess)}
      onAllowSiteAccess={requestSiteAccess}
      status={status}
      model={model}
      onLoadModel={requestWarmUp}
      onDownloadModel={requestDownload}
      onCancelDownload={requestCancelDownload}
    />
  );
}
