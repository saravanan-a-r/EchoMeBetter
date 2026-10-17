import { canRewrite } from '../../shared/modelInstall';
import { shortcutLabel, type KeyPlatform } from '../../shared/shortcuts';
import type { EngineStatus } from '../../shared/status';
import type { StyleId } from '../../shared/styles';
import { Logo, Wordmark } from '../brand/Logo';
import { PulseLogo } from '../brand/PulseLogo';
import { ModelCard } from '../components/ModelCard';
import { PerformanceSettings, type PerformanceSettingsProps } from '../components/PerformanceSettings';
import { ShortcutAccessCard } from '../components/ShortcutAccessCard';
import { PopupStyles } from '../components/PopupStyles';
import { StyleList } from '../components/StyleList';
import { MACHINE, retryGpu, useGpu, useGpuProblem } from '../hooks/useCompute';
import { requestWarmUp, useEngineStatus } from '../hooks/useEngineStatus';
import {
  modelSetup,
  requestCancelDownload,
  requestDownload,
  useCatalog,
  useModelAvailability,
  type CatalogState,
  type ModelAvailability,
} from '../hooks/useModel';
import { useSettings } from '../hooks/useSettings';
import { needsSiteAccess, PLATFORM, requestSiteAccess, websiteAvailability, useSiteAccess, type WebsiteAvailability } from '../hooks/useShortcuts';
import { WaysToRewrite } from './WaysToRewrite';

export const PLAYGROUND_SAMPLE =
  "hey team, just wanted to let u know the release is gonna slip to next week because QA found a couple of bugs in the checkout flow. sorry about that, we're on it";

export const WELCOME_HOLD_HINT = 'Or press and hold on the selected text for 1 second to choose a style.';

const ICON = { width: 20, height: 20, viewBox: '0 0 16 16', fill: 'none', stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round' } as const;

/** What the header promises, as plain facts with an icon each (not controls): a shield, a person crossed out, a wifi signal crossed out. */
const TRUST = [
  {
    label: 'Runs on your device',
    icon: (
      <svg {...ICON}>
        <path d="M8 1.5 2.5 3.8v3.6c0 3.3 2.3 6.1 5.5 7.1 3.2-1 5.5-3.8 5.5-7.1V3.8L8 1.5Z" />
        <path d="m5.6 8 1.7 1.7 3.2-3.3" />
      </svg>
    ),
  },
  {
    label: 'No account required',
    icon: (
      <svg {...ICON}>
        <circle cx="8" cy="5.5" r="2.8" />
        <path d="M2.8 14c.6-2.6 2.7-4.2 5.2-4.2s4.6 1.6 5.2 4.2M2 2l12 12" />
      </svg>
    ),
  },
  {
    label: 'Works offline after setup',
    icon: (
      <svg {...ICON}>
        <path d="M2 6.2a8.5 8.5 0 0 1 12 0M4.3 8.6a5.2 5.2 0 0 1 7.4 0M6.6 11a2 2 0 0 1 2.8 0M2 2l12 12" />
      </svg>
    ),
  },
];

export interface WelcomeViewProps {
  readonly status: EngineStatus;
  readonly model: ModelAvailability;
  readonly catalog: CatalogState;
  readonly onLoadModel: () => void;
  readonly onDownload: (adapters: StyleId[]) => void;
  readonly onCancelDownload: () => void;
  readonly onRetryCatalog: () => void;
  readonly platform: KeyPlatform;
  readonly shortcuts: WebsiteAvailability;
  readonly holdMenu: WebsiteAvailability;
  readonly onAllowSiteAccess: () => void;
  /** Null until the settings have been read. */
  readonly performance?: PerformanceSettingsProps | null;
}

export function WelcomeView(props: WelcomeViewProps) {
  const { model, platform, shortcuts, holdMenu, onAllowSiteAccess } = props;
  const setup = modelSetup(model.installed, props.catalog);
  // This page listens for shortcuts and holds itself, so they work here even before websites are allowed.
  const showKeys = shortcuts === 'on' || shortcuts === 'needs-access' ? platform : undefined;
  const showHold = holdMenu === 'on' || holdMenu === 'needs-access';
  return (
    <div className="_echo_$_min-h-screen _echo_$_bg-canvas _echo_$_text-fg">
      <header className="_echo_$_relative _echo_$_theme-inverse _echo_$_overflow-hidden _echo_$_bg-canvas _echo_$_text-fg">
        <div aria-hidden="true" className="_echo_$_absolute _echo_$_inset-0 _echo_$_bg-echo-glow" />
        <div className="_echo_$_relative _echo_$_mx-auto _echo_$_flex _echo_$_max-w-5xl _echo_$_flex-col _echo_$_items-center _echo_$_px-6 _echo_$_pb-16 _echo_$_pt-14 _echo_$_text-center">
          <PulseLogo size={72} />
          <h1 className="_echo_$_mt-6 _echo_$_text-4xl _echo_$_font-bold _echo_$_tracking-tight sm:_echo_$_text-5xl">
            Rewrite anything, <span className="_echo_$_bg-echo-gradient _echo_$_bg-clip-text _echo_$_text-transparent">right where you type.</span>
          </h1>
          <p className="_echo_$_mt-4 _echo_$_max-w-2xl _echo_$_text-lg _echo_$_text-fg-soft">
            <Wordmark /> rewrites the text you select in any text box to be more professional, grammatically correct, friendlier, more
            concise or more detailed, using a writing model that runs entirely on your device.
          </p>
          <ul className="_echo_$_mt-10 _echo_$_grid _echo_$_w-full _echo_$_max-w-3xl _echo_$_list-none _echo_$_gap-3 _echo_$_border-0 _echo_$_border-t _echo_$_border-solid _echo_$_border-sheen/15 _echo_$_p-0 _echo_$_pt-6 sm:_echo_$_grid-cols-3 sm:_echo_$_gap-0 sm:_echo_$_divide-x sm:_echo_$_divide-solid sm:_echo_$_divide-sheen/15">
            {TRUST.map(({ label, icon }) => (
              <li key={label} className="_echo_$_flex _echo_$_items-center _echo_$_justify-center _echo_$_gap-2.5 _echo_$_px-4 _echo_$_text-[15px] _echo_$_text-fg-soft">
                <span aria-hidden="true" className="_echo_$_text-better-300">
                  {icon}
                </span>
                {label}
              </li>
            ))}
          </ul>
        </div>
      </header>

      <main className="_echo_$_mx-auto _echo_$_max-w-5xl _echo_$_space-y-14 _echo_$_px-6 _echo_$_py-12">
        <WaysToRewrite platform={showKeys} hold={showHold} />

        <section aria-labelledby="try-title" className="_echo_$_rounded-3xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_p-6 _echo_$_shadow-card">
          <div className="_echo_$_flex _echo_$_flex-wrap _echo_$_items-start _echo_$_justify-between _echo_$_gap-4">
            <div>
              <h2 id="try-title" className="_echo_$_text-2xl _echo_$_font-bold">Try it here</h2>
              <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_text-fg-muted">
                {!model.known || canRewrite(model.installed)
                  ? 'Select some of the text below, right-click and choose EchoMeBetter.'
                  : model.installed
                    ? 'Download a style, then select text below, right-click and choose EchoMeBetter.'
                    : 'Download the model, then select text below, right-click and choose EchoMeBetter.'}
              </p>
              {showHold ? (
                <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_text-fg-muted">{WELCOME_HOLD_HINT}</p>
              ) : null}
              {showKeys ? (
                <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_text-fg-muted">
                  Or use the keyboard: select text and press {shortcutLabel('professional', showKeys)} for Professional.
                </p>
              ) : null}
            </div>
            <div className="_echo_$_w-full _echo_$_space-y-3.5 sm:_echo_$_w-80">
              <ModelCard
                status={props.status}
                model={model}
                catalog={props.catalog}
                onLoadModel={props.onLoadModel}
                onDownload={props.onDownload}
                onCancelDownload={props.onCancelDownload}
                onRetryCatalog={props.onRetryCatalog}
              />
              {/* The same styles list as the popup: it follows the setup, and each style not yet here is a download icon. */}
              {model.installed && setup.catalog ? (
                <PopupStyles
                  readiness={setup.readiness}
                  onDownload={(adapter) => props.onDownload([adapter])}
                  busy={model.download.state === 'downloading'}
                  downloading={model.download.state === 'downloading' ? model.download.target.adapters : []}
                />
              ) : null}
            </div>
          </div>
          {needsSiteAccess(shortcuts, holdMenu) ? (
            <div className="_echo_$_mt-5">
              <ShortcutAccessCard
                platform={platform}
                waiting={{ shortcuts: shortcuts === 'needs-access', holdMenu: holdMenu === 'needs-access' }}
                onAllow={onAllowSiteAccess}
              />
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
            className="_echo_$_mt-5 _echo_$_w-full _echo_$_resize-y _echo_$_rounded-2xl _echo_$_border _echo_$_border-line-strong _echo_$_bg-canvas _echo_$_p-4 _echo_$_text-base _echo_$_leading-7 _echo_$_text-fg _echo_$_outline-none focus:_echo_$_border-accent-fg focus:_echo_$_ring-4 focus:_echo_$_ring-accent-subtle"
          />
        </section>

        <section aria-labelledby="styles-title">
          <h2 id="styles-title" className="_echo_$_text-2xl _echo_$_font-bold">Five writing styles</h2>
          <div className="_echo_$_mt-6">
            <StyleList columns={3} shortcuts={showKeys} />
          </div>
        </section>

        {props.performance && props.performance.gpu.state !== 'checking' ? (
          <section aria-labelledby="performance-title" className="_echo_$_grid _echo_$_items-start _echo_$_gap-6 md:_echo_$_grid-cols-[1fr_minmax(0,24rem)]">
            <div>
              <h2 id="performance-title" className="_echo_$_text-2xl _echo_$_font-bold">
                Speed and power
              </h2>
              <p className="_echo_$_mt-2 _echo_$_text-sm _echo_$_leading-6 _echo_$_text-fg-muted">
                {props.performance.gpu.state === 'available'
                  ? 'EchoMeBetter runs on your graphics chip (GPU), the fastest way to rewrite. Rather keep it on the processor, or save battery? Choose here, or later in the settings of the toolbar popup.'
                  : 'EchoMeBetter runs on your processor. Choose how much of it a rewrite may use, here or later in the settings of the toolbar popup.'}
              </p>
            </div>
            <div className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_p-5">
              <PerformanceSettings {...props.performance} />
            </div>
          </section>
        ) : null}

        <section className="_echo_$_grid _echo_$_gap-4 md:_echo_$_grid-cols-3">
          {[
            ['Private by design', 'The model runs inside your browser. Your text is never uploaded, logged or shared.'],
            ['Lightweight', 'Rewrites run in a background worker, so the page you are typing in stays responsive. The model unloads automatically when idle.'],
            ['Works where you write', 'Text areas, input fields and rich-text editors such as email composers. Native undo keeps working.'],
          ].map(([title, body]) => (
            <div key={title} className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_p-5">
              <h3 className="_echo_$_font-semibold">{title}</h3>
              <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_leading-6 _echo_$_text-fg-muted">{body}</p>
            </div>
          ))}
        </section>
      </main>

      <footer className="_echo_$_flex _echo_$_items-center _echo_$_justify-center _echo_$_gap-2 _echo_$_border-0 _echo_$_border-t _echo_$_border-solid _echo_$_border-line _echo_$_px-6 _echo_$_py-8 _echo_$_text-center _echo_$_text-sm _echo_$_text-fg-muted">
        <span className="_echo_$_shrink-0">
          <Logo size={18} />
        </span>
        <span>
          <Wordmark /> · Everything runs on your device. Nothing you write leaves your browser.
        </span>
      </footer>
    </div>
  );
}

export function WelcomeApp() {
  const status = useEngineStatus();
  const model = useModelAvailability();
  const [catalog, retryCatalog] = useCatalog();
  const [settings, setSettings, settingsLoaded] = useSettings();
  const siteAccess = useSiteAccess();
  const gpu = useGpu();
  const gpuProblem = useGpuProblem();
  return (
    <WelcomeView
      performance={
        settingsLoaded
          ? {
              compute: settings,
              gpu,
              gpuProblem,
              cores: MACHINE.cores,
              crossOriginIsolated: MACHINE.crossOriginIsolated,
              status,
              onChange: (compute) => setSettings({ ...settings, ...compute }),
              onRetryGpu: retryGpu,
            }
          : null
      }
      platform={PLATFORM}
      shortcuts={websiteAvailability(settingsLoaded ? settings.shortcutsEnabled : null, siteAccess)}
      holdMenu={websiteAvailability(settingsLoaded ? settings.holdMenuEnabled : null, siteAccess)}
      onAllowSiteAccess={requestSiteAccess}
      status={status}
      model={model}
      catalog={catalog}
      onLoadModel={requestWarmUp}
      onDownload={requestDownload}
      onCancelDownload={requestCancelDownload}
      onRetryCatalog={retryCatalog}
    />
  );
}
