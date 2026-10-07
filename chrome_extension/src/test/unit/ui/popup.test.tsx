/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { DEFAULT_COMPUTE, type ComputeSettings } from '../../../shared/compute';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { formatBytes, formatDuration, formatList, formatTimeLeft } from '../../../shared/format';
import type { UiReply } from '../../../shared/messages';
import type { DownloadState, InstalledModelRecord, ModelCatalog } from '../../../shared/modelInstall';
import { MODEL_SOURCE_URL } from '../../../shared/modelSource';
import type { EngineStatus, ModelSummary } from '../../../shared/status';
import type { StyleId } from '../../../shared/styles';
import { describeTarget, presentDownload } from '../../../ui/components/ModelDownloadCard';
import { GPU_PROBLEM_NOTE, PerformanceSettings, type PerformanceSettingsProps } from '../../../ui/components/PerformanceSettings';
import { presentStatus } from '../../../ui/components/StatusCard';
import { ACCESS_TITLE } from '../../../ui/components/ShortcutAccessCard';
import type { CatalogState, ModelAvailability } from '../../../ui/hooks/useModel';
import { shortcutAvailability, type ShortcutAvailability } from '../../../ui/hooks/useShortcuts';
import { SettingsView, SHORTCUT_NOTE } from '../../../ui/popup/SettingsView';
import { PopupView, REMOVED_NOTICE } from '../../../ui/popup/PopupApp';
import { WelcomeView } from '../../../ui/welcome/WelcomeApp';

const summary: ModelSummary = { id: 'echomebetter-int8', displayName: 'EchoMeBetter', precision: 'int8', sizeBytes: 790_000_000 };
/** What a model.json offers: adapters for `adapters`, every other style falling back to professional. */
const catalogWith = (adapters: StyleId[]): ModelCatalog => ({
  base: 'base-1',
  model: summary,
  adapters: adapters.map((style) => ({ style, sizeBytes: 73_000_000 })),
  fallbackAdapter: 'professional',
});
/** Today's package: only the professional adapter is trained. */
const TODAY = catalogWith(['professional']);
const served = (catalog: ModelCatalog = TODAY): CatalogState => ({ status: 'ready', catalog });
const record = (adapters: StyleId[] = ['professional'], catalog = TODAY): InstalledModelRecord => ({ sourceUrl: MODEL_SOURCE_URL, catalog, adapters });
const installed = (adapters?: StyleId[], download: DownloadState = { state: 'idle' }, catalog?: ModelCatalog): ModelAvailability => ({
  known: true,
  installed: record(adapters, catalog),
  download,
});
const missing = (download: DownloadState = { state: 'idle' }): ModelAvailability => ({ known: true, installed: null, download });
const FIRST = { base: true, adapters: ['professional'] } as const;

function renderPopup(
  status: EngineStatus,
  model: ModelAvailability = installed(),
  options: { modelRemoved?: boolean; shortcuts?: ShortcutAvailability; catalog?: CatalogState } = {},
) {
  const handlers = {
    onAllowSiteAccess: jest.fn(),
    onTurnOffShortcuts: jest.fn(),
    onOpenSettings: jest.fn(),
    onLoadModel: jest.fn(),
    onOpenGuide: jest.fn(),
    onDownload: jest.fn(),
    onCancelDownload: jest.fn(),
    onRetryCatalog: jest.fn(),
  };
  render(
    <PopupView
      status={status}
      model={model}
      catalog={options.catalog ?? served()}
      modelRemoved={options.modelRemoved ?? false}
      platform="mac"
      shortcuts={options.shortcuts ?? 'off'}
      {...handlers}
    />,
  );
  return handlers;
}

const ready: EngineStatus = { state: 'ready', model: summary, runningOn: { processor: 'gpu', threads: 4 } };
const row = (label: string) => screen.getByText(label, { selector: 'span' }).closest('li')!;

describe('popup with the model downloaded', () => {
  test('sleeping model offers to load now', () => {
    const { onLoadModel } = renderPopup({ state: 'unloaded' });
    fireEvent.click(screen.getByRole('button', { name: 'Load now' }));
    expect(onLoadModel).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: /^Download ·/ })).not.toBeInTheDocument();
  });

  test('loading shows progress', () => {
    renderPopup({ state: 'loading', progress: 0.37 });
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '37');
    expect(screen.getByText('Loading · 37%')).toBeInTheDocument();
  });

  test('ready names the model and what it runs on', () => {
    renderPopup(ready);
    expect(screen.getByText('EchoMeBetter · int8 · 790 MB · on-device GPU')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
  });

  test('an error can be retried', () => {
    const { onLoadModel } = renderPopup({ state: 'error', message: 'out of memory' });
    expect(screen.getByText('out of memory')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(onLoadModel).toHaveBeenCalled();
  });

  test('every style is listed; those still in training say they run on Professional for now', () => {
    renderPopup(ready);
    expect(within(row('Professional')).queryByText('Preview')).not.toBeInTheDocument();
    for (const label of ['Grammar', 'Friendly', 'Concise', 'Elaborate']) {
      expect(within(row(label)).getByText('Preview')).toHaveAttribute(
        'title',
        `${label}'s own style is still in training. Until it's ready, ${label} rewrites with the Professional style.`,
      );
    }
  });

  test('a style whose own adapter is out but not downloaded offers it', () => {
    const both = catalogWith(['professional', 'grammar']);
    const { onDownload } = renderPopup(ready, installed(['professional'], { state: 'idle' }, both), { catalog: served(both) });
    expect(within(row('Grammar')).queryByText('Preview')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download the Grammar style (73 MB)' }));
    expect(onDownload).toHaveBeenCalledWith(['grammar']);
  });

  test('adapters published after the base was installed are offered from the server', () => {
    renderPopup(ready, installed(['professional']), { catalog: served(catalogWith(['professional', 'grammar'])) });
    expect(screen.getByRole('button', { name: 'Download the Grammar style (73 MB)' })).toBeInTheDocument();
  });

  test('without the adapter a style runs on, the style says what it needs', () => {
    renderPopup(ready, installed([]));
    expect(screen.getByRole('button', { name: 'Download the Professional style (73 MB)' })).toHaveTextContent('Get · 73 MB');
    for (const label of ['Grammar', 'Friendly', 'Concise', 'Elaborate']) expect(within(row(label)).getByText('Needs Professional')).toBeInTheDocument();
  });

  test('a style download shows its progress in place of the status, and no other can start meanwhile', () => {
    renderPopup(
      ready,
      installed([], {
        state: 'downloading',
        target: { base: false, adapters: ['professional'] },
        phase: 'fetching',
        receivedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: 0,
      }),
    );
    expect(screen.getByText('Professional style')).toBeInTheDocument();
    expect(screen.queryByText('Ready')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Download the Professional style (73 MB)' })).toBeDisabled();
  });
});

describe('keyboard shortcuts in the popup', () => {
  test('the gear opens settings', () => {
    const { onOpenSettings } = renderPopup(ready);
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  test('working shortcuts are listed next to their styles, for this platform', () => {
    renderPopup(ready, installed(), { shortcuts: 'on' });
    expect(screen.getByLabelText('Control Shift P')).toHaveTextContent('⌃⇧P');
    expect(screen.queryByText(ACCESS_TITLE)).not.toBeInTheDocument();
  });

  test('turned off, no keys are shown and nothing is asked', () => {
    renderPopup(ready, installed(), { shortcuts: 'off' });
    expect(screen.queryByLabelText(/Shift/)).not.toBeInTheDocument();
    expect(screen.getByText('Polished and workplace-ready')).toBeInTheDocument();
    expect(screen.queryByText(ACCESS_TITLE)).not.toBeInTheDocument();
  });

  test('without site access the popup asks, and can be told not now', () => {
    const { onAllowSiteAccess, onTurnOffShortcuts } = renderPopup(ready, installed(), { shortcuts: 'needs-access' });
    expect(screen.getByText(ACCESS_TITLE)).toBeInTheDocument();
    expect(screen.getByText(/shortcuts like ⌃⇧P on websites/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Shift/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    expect(onAllowSiteAccess).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(onTurnOffShortcuts).toHaveBeenCalledTimes(1);
  });
});

function renderSettings(
  shortcuts: ShortcutAvailability,
  options: { platform?: 'mac' | 'other'; installed?: InstalledModelRecord | null; removeReply?: UiReply; performance?: PerformanceSettingsProps } = {},
) {
  const handlers = {
    onShortcutsChange: jest.fn(),
    onAllowSiteAccess: jest.fn(),
    onKeepLoadedChange: jest.fn(),
    onBack: jest.fn(),
    onRemoveModel: jest.fn(async (): Promise<UiReply> => options.removeReply ?? { ok: true }),
    onRemoveAdapter: jest.fn(async (_adapter: StyleId): Promise<UiReply> => options.removeReply ?? { ok: true }),
  };
  render(
    <SettingsView
      platform={options.platform ?? 'other'}
      shortcuts={shortcuts}
      keepLoaded={15}
      installed={options.installed ?? null}
      performance={options.performance ?? null}
      {...handlers}
    />,
  );
  return handlers;
}

describe('settings page', () => {
  test('shortcuts on: the switch is on and every shortcut is listed', () => {
    const { onShortcutsChange } = renderSettings('on');
    const toggle = screen.getByRole('switch', { name: 'Keyboard shortcuts' });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    for (const letter of ['P', 'G', 'F', 'C', 'E']) expect(screen.getByRole('img', { name: `Alt Shift ${letter}` })).toBeInTheDocument();
    expect(screen.getByText(SHORTCUT_NOTE)).toBeInTheDocument();
    fireEvent.click(toggle);
    expect(onShortcutsChange).toHaveBeenCalledWith(false);
  });

  test('shortcuts off: switching on asks for them', () => {
    const { onShortcutsChange } = renderSettings('off');
    expect(screen.getByRole('switch', { name: 'Keyboard shortcuts' })).toHaveAttribute('aria-checked', 'false');
    expect(screen.queryByRole('list', { name: 'Shortcuts' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('switch', { name: 'Keyboard shortcuts' }));
    expect(onShortcutsChange).toHaveBeenCalledWith(true);
  });

  test('on but not allowed on websites: explains and offers to allow', () => {
    const { onAllowSiteAccess } = renderSettings('needs-access', { platform: 'mac' });
    expect(screen.getByText('One more step: allow on websites')).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: 'Keyboard shortcuts' })).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    expect(onAllowSiteAccess).toHaveBeenCalledTimes(1);
  });

  test('nothing is switchable before the state is known', () => {
    renderSettings('loading');
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
  });

  test('the idle setting reports a number of minutes, and back returns', () => {
    const { onKeepLoadedChange, onBack } = renderSettings('on');
    fireEvent.change(screen.getByLabelText(/Free memory when idle/), { target: { value: '0' } });
    expect(onKeepLoadedChange).toHaveBeenCalledWith(0);
    fireEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(onBack).toHaveBeenCalled();
  });

  test('availability combines the setting with the site access', () => {
    expect(shortcutAvailability(null, true)).toBe('loading');
    expect(shortcutAvailability(true, null)).toBe('loading');
    expect(shortcutAvailability(false, true)).toBe('off');
    expect(shortcutAvailability(true, false)).toBe('needs-access');
    expect(shortcutAvailability(true, true)).toBe('on');
  });
});

function performance(overrides: Partial<PerformanceSettingsProps> = {}): PerformanceSettingsProps {
  return {
    compute: DEFAULT_COMPUTE,
    gpu: { state: 'available', gpu: { vendor: 'apple' } },
    gpuProblem: null,
    cores: 10,
    crossOriginIsolated: true,
    status: { state: 'unloaded' },
    onChange: jest.fn(),
    onRetryGpu: jest.fn(),
    ...overrides,
  };
}

function renderPerformance(overrides: Partial<PerformanceSettingsProps> = {}) {
  const props = performance(overrides);
  render(<PerformanceSettings {...props} />);
  return props;
}

const withCompute = (compute: Partial<ComputeSettings>) => ({ compute: { ...DEFAULT_COMPUTE, ...compute } });

describe('performance settings', () => {
  test('with a GPU on offer it is the default, named, with its power setting', () => {
    const { onChange } = renderPerformance();
    expect(screen.getByRole('radiogroup', { name: 'Run the model on' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /GPU/ })).toBeChecked();
    expect(screen.getByText(/run on your Apple graphics chip/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Processor use')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('GPU power'), { target: { value: 'low-power' } });
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_COMPUTE, gpuPower: 'low-power' });
    fireEvent.click(screen.getByRole('radio', { name: /CPU/ }));
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_COMPUTE, processor: 'cpu' });
  });

  test('on the processor, its share is offered in cores of this computer', () => {
    const { onChange } = renderPerformance(withCompute({ processor: 'cpu' }));
    expect(screen.getByRole('radio', { name: /CPU/ })).toBeChecked();
    expect(screen.queryByLabelText('GPU power')).not.toBeInTheDocument();
    const usage = screen.getByLabelText('Processor use');
    expect(within(usage).getAllByRole('option').map((option) => option.textContent)).toEqual(['Light · 2 cores', 'Balanced · 4 cores', 'Maximum · 6 cores']);
    expect(usage).toHaveValue('balanced');
    expect(screen.getByText(/How much of the processor \(10 cores\) a rewrite may use/)).toBeInTheDocument();
    fireEvent.change(usage, { target: { value: 'maximum' } });
    expect(onChange).toHaveBeenCalledWith({ ...DEFAULT_COMPUTE, processor: 'cpu', cpuUsage: 'maximum' });
  });

  test('without a GPU there is no GPU choice at all, only the processor share', () => {
    renderPerformance({ gpu: { state: 'none' } });
    expect(screen.queryByRole('radiogroup')).not.toBeInTheDocument();
    expect(screen.queryByText(/GPU/)).not.toBeInTheDocument();
    expect(screen.getByLabelText('Processor use')).toBeInTheDocument();
  });

  test('nothing is shown until it is known whether there is a GPU', () => {
    const { container } = render(<PerformanceSettings {...performance({ gpu: { state: 'checking' } })} />);
    expect(container).toBeEmptyDOMElement();
  });

  test('a GPU that could not run the model is explained, can be tried again, and the processor share applies meanwhile', () => {
    const { onRetryGpu } = renderPerformance({ gpuProblem: { message: 'buffer too large' } });
    expect(screen.getByRole('status')).toHaveTextContent(GPU_PROBLEM_NOTE);
    expect(screen.getByLabelText('Processor use')).toBeInTheDocument();
    expect(screen.queryByLabelText('GPU power')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try the GPU again' }));
    expect(onRetryGpu).toHaveBeenCalledTimes(1);
  });

  test('says where the model runs right now', () => {
    renderPerformance({ status: ready });
    expect(screen.getByText('Running on the GPU now.')).toBeInTheDocument();
  });

  test.each([
    [{ state: 'ready', model: summary, runningOn: { processor: 'cpu', threads: 4 } } as EngineStatus, 'Running on the processor now, with 4 threads.'],
    [{ state: 'loading', progress: 0.2 } as EngineStatus, 'Loading the model…'],
  ])('status line: %j', (status, text) => {
    renderPerformance({ status });
    expect(screen.getByText(text)).toBeInTheDocument();
  });

  test('the settings page shows the section once the settings and the GPU are known', () => {
    renderSettings('off', { performance: performance() });
    expect(within(screen.getByRole('region', { name: 'Performance' })).getByRole('radio', { name: /GPU/ })).toBeChecked();
  });

  test('the settings page leaves it out while the settings are being read', () => {
    renderSettings('off');
    expect(screen.queryByRole('region', { name: 'Performance' })).not.toBeInTheDocument();
  });
});

describe('storage settings', () => {
  test('lists the writing model and each downloaded style with its size', () => {
    renderSettings('off', { installed: record(['professional', 'grammar'], catalogWith(['professional', 'grammar'])) });
    const items = within(screen.getByRole('list', { name: 'Downloaded' })).getAllByRole('listitem');
    expect(items.map((item) => item.textContent)).toEqual(['Writing model790 MBRemove', 'Professional style73 MBRemove', 'Grammar style73 MBRemove']);
  });

  test('says so when nothing is downloaded', () => {
    renderSettings('off');
    expect(screen.getByText('Nothing downloaded yet.')).toBeInTheDocument();
  });

  test('removing a style asks first and names every style that stops working', async () => {
    const { onRemoveAdapter } = renderSettings('off', { installed: record(['professional']) });
    fireEvent.click(screen.getByRole('button', { name: 'Remove professional style' }));
    expect(screen.getByText('Remove the Professional style?')).toBeInTheDocument();
    expect(
      screen.getByText("This frees about 73 MB. Professional, Grammar, Friendly, Concise and Elaborate can't be used until you download it again."),
    ).toBeInTheDocument();
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Remove' })));
    expect(onRemoveAdapter).toHaveBeenCalledWith('professional');
  });

  test('removing the writing model takes every style with it, and can be kept', () => {
    const { onRemoveModel } = renderSettings('off', { installed: record(['professional']) });
    fireEvent.click(screen.getByRole('button', { name: 'Remove writing model' }));
    expect(screen.getByText('Remove the writing model and every style?')).toBeInTheDocument();
    expect(
      screen.getByText("This frees about 863 MB of disk space. EchoMeBetter can't rewrite text until you download it again. Your settings are kept."),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Keep it' }));
    expect(onRemoveModel).not.toHaveBeenCalled();
    expect(screen.queryByText('Remove the writing model and every style?')).not.toBeInTheDocument();
  });

  test('a rewrite in progress blocks a removal, with a reason', async () => {
    renderSettings('off', { installed: record(['professional']), removeReply: { ok: false, error: { code: 'BUSY' } } });
    fireEvent.click(screen.getByRole('button', { name: 'Remove writing model' }));
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Remove' })));
    expect(screen.getByRole('alert')).toHaveTextContent('Wait for the current rewrite to finish.');
    expect(screen.getByRole('button', { name: 'Remove' })).toBeEnabled();
  });
});

describe('popup without the model', () => {
  test('lists what can be downloaded, everything chosen, and shows where it comes from', () => {
    const { onDownload } = renderPopup({ state: 'unloaded' }, missing());
    expect(screen.getByText('Writing model needed')).toBeInTheDocument();
    expect(screen.getByText(MODEL_SOURCE_URL)).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: /Writing model/ })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: /Writing model/ })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: /Professional style/ })).toBeChecked();
    expect(screen.getByText('Also used by Grammar, Friendly, Concise and Elaborate for now')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download · 863 MB' }));
    expect(onDownload).toHaveBeenCalledWith(['professional']);
  });

  test('the writing model can be downloaded on its own', () => {
    const { onDownload } = renderPopup({ state: 'unloaded' }, missing());
    fireEvent.click(screen.getByRole('checkbox', { name: /Professional style/ }));
    expect(screen.getByText(/Without a style there is nothing to rewrite with yet/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download · 790 MB' }));
    expect(onDownload).toHaveBeenCalledWith([]);
  });

  test('while the server is being asked, nothing can be chosen yet', () => {
    renderPopup({ state: 'unloaded' }, missing(), { catalog: { status: 'loading' } });
    expect(screen.getByText("Checking what's available…")).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Download ·/ })).not.toBeInTheDocument();
  });

  test('an unreachable server can be asked again', () => {
    const { onRetryCatalog } = renderPopup({ state: 'unloaded' }, missing(), { catalog: { status: 'failed' } });
    expect(screen.getByRole('alert')).toHaveTextContent("Couldn't reach the download server.");
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(onRetryCatalog).toHaveBeenCalledTimes(1);
  });

  test('shows what is downloading, the progress, speed and time left', () => {
    renderPopup(
      { state: 'unloaded' },
      missing({ state: 'downloading', target: FIRST, phase: 'fetching', receivedBytes: 312_000_000, totalBytes: 863_000_000, bytesPerSecond: 24_000_000 }),
    );
    expect(screen.getByText('Downloading · 36%')).toBeInTheDocument();
    expect(screen.getByText('Writing model and Professional style')).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Download progress' })).toHaveAttribute('aria-valuenow', '36');
    expect(screen.getByText('312 MB of 863 MB · 24 MB/s · about 23 s left')).toBeInTheDocument();
    expect(screen.getByText('You can close this window. The download keeps going.')).toBeInTheDocument();
  });

  test('cancelling asks first, and can be backed out of', () => {
    const { onCancelDownload } = renderPopup(
      { state: 'unloaded' },
      missing({ state: 'downloading', target: FIRST, phase: 'fetching', receivedBytes: 1, totalBytes: 10, bytesPerSecond: 0 }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Cancel download' }));
    expect(screen.getByText('Stop and delete the partly downloaded files?')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Keep downloading' }));
    expect(onCancelDownload).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Cancel download' }));
    fireEvent.click(screen.getByRole('button', { name: 'Stop download' }));
    expect(onCancelDownload).toHaveBeenCalledTimes(1);
  });

  test('a stopped download explains why, can be tried again as it was, or put aside', () => {
    const { onDownload, onCancelDownload } = renderPopup(
      { state: 'unloaded' },
      missing({ state: 'failed', target: FIRST, error: { code: 'STORAGE_FULL', details: { neededBytes: 300_000_000 } } }),
    );
    expect(screen.getByText('Download stopped')).toBeInTheDocument();
    expect(screen.getByText(/Free up 300 MB and try again/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(onDownload).toHaveBeenCalledWith(['professional']);
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(onCancelDownload).toHaveBeenCalledTimes(1);
  });

  test('after removing the model the popup says so', () => {
    renderPopup({ state: 'unloaded' }, missing(), { modelRemoved: true });
    expect(screen.getByText(REMOVED_NOTICE)).toBeInTheDocument();
  });

  test('nothing model-related renders before storage has been read', () => {
    renderPopup({ state: 'unloaded' }, { known: false, installed: null, download: { state: 'idle' } });
    expect(screen.queryByRole('button', { name: /^Download ·/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
  });
});

describe('welcome page', () => {
  const props = {
    status: { state: 'unloaded' } as EngineStatus,
    catalog: served(),
    onLoadModel: jest.fn(),
    onDownload: jest.fn(),
    onCancelDownload: jest.fn(),
    onRetryCatalog: jest.fn(),
    platform: 'other' as const,
    shortcuts: 'off' as ShortcutAvailability,
    onAllowSiteAccess: jest.fn(),
    performance: null,
  };

  test('offers the download next to the practice box until the model is there', () => {
    render(<WelcomeView {...props} model={missing()} />);
    expect(screen.getByText(/Download the model, then select text below/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download · 863 MB' }));
    expect(props.onDownload).toHaveBeenCalledWith(['professional']);
  });

  test('shows the model status once downloaded', () => {
    render(<WelcomeView {...props} model={installed()} />);
    expect(screen.getByRole('button', { name: 'Load now' })).toBeInTheDocument();
    expect(screen.getByText(/Select some of the text below/)).toBeInTheDocument();
  });

  test('with the model but no style, a style is what is still needed', () => {
    render(<WelcomeView {...props} model={installed([])} />);
    expect(screen.getByText(/Download a style, then select text below/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download the Professional style (73 MB)' }));
    expect(props.onDownload).toHaveBeenCalledWith(['professional']);
  });

  test('teaches the shortcuts while they are on, and asks for site access when needed', () => {
    render(<WelcomeView {...props} model={installed()} shortcuts="needs-access" />);
    expect(screen.getByText(/press Alt\+Shift\+P for Professional/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    expect(props.onAllowSiteAccess).toHaveBeenCalled();
  });

  test('lets the user choose GPU or CPU where a GPU is on offer, saying where to change it later', () => {
    render(<WelcomeView {...props} model={installed()} performance={performance()} />);
    const section = screen.getByRole('region', { name: 'Speed and power' });
    expect(within(section).getByText(/runs on your graphics chip \(GPU\)/)).toBeInTheDocument();
    expect(within(section).getByText(/settings of the toolbar popup/)).toBeInTheDocument();
    expect(within(section).getByRole('radio', { name: /GPU/ })).toBeChecked();
  });

  test('without a GPU it speaks only of the processor', () => {
    render(<WelcomeView {...props} model={installed()} performance={performance({ gpu: { state: 'none' } })} />);
    const section = screen.getByRole('region', { name: 'Speed and power' });
    expect(within(section).queryByText(/GPU|graphics/)).not.toBeInTheDocument();
    expect(within(section).getByLabelText('Processor use')).toBeInTheDocument();
  });

  test('with shortcuts off it only teaches the right-click', () => {
    render(<WelcomeView {...props} model={installed()} />);
    expect(screen.queryByText(/press Alt\+Shift/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Allow on websites' })).not.toBeInTheDocument();
  });
});

test('status and download presentation', () => {
  expect(presentStatus({ state: 'unloaded' }).action).toBe('Load now');
  expect(presentDownload({ state: 'downloading', target: FIRST, phase: 'fetching', receivedBytes: 0, totalBytes: 0, bytesPerSecond: 0 })).toEqual({
    label: 'Downloading',
    detail: 'Connecting…',
    percent: 0,
  });
  expect(presentDownload({ state: 'downloading', target: FIRST, phase: 'verifying', receivedBytes: 9, totalBytes: 9, bytesPerSecond: 0 }).label).toBe(
    'Checking files…',
  );
  expect(describeTarget({ base: true, adapters: [] })).toBe('Writing model');
  expect(describeTarget({ base: false, adapters: ['professional', 'grammar'] })).toBe('Professional and Grammar styles');
});

test('sizes, times and lists', () => {
  expect(formatBytes(1_500_000_000)).toBe('1.5 GB');
  expect(formatBytes(820_000_000)).toBe('820 MB');
  expect(formatBytes(512_000)).toBe('512 KB');
  expect(formatTimeLeft(21.4)).toBe('about 21 s left');
  expect(formatTimeLeft(150)).toBe('about 3 min left');
  expect(formatTimeLeft(3900)).toBe('about 1 h 5 min left');
  expect(formatList(['A'])).toBe('A');
  expect(formatList(['A', 'B', 'C'])).toBe('A, B and C');
});

test('rewrite durations', () => {
  expect(formatDuration(420)).toBe('420 ms');
  expect(formatDuration(999.4)).toBe('999 ms');
  expect(formatDuration(1000)).toBe('1.0 s');
  expect(formatDuration(2345)).toBe('2.3 s');
  expect(formatDuration(60_000)).toBe('1 min');
  expect(formatDuration(95_000)).toBe('1 min 35 s');
});
