/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { formatBytes, formatDuration, formatList, formatTimeLeft } from '../../../shared/format';
import type { UiReply } from '../../../shared/messages';
import type { DownloadState, InstalledModelRecord, ModelCatalog } from '../../../shared/modelInstall';
import { MODEL_SOURCE_URL } from '../../../shared/modelSource';
import type { EngineStatus, ModelSummary } from '../../../shared/status';
import type { StyleId } from '../../../shared/styles';
import { describeTarget, presentDownload } from '../../../ui/components/ModelDownloadCard';
import { presentStatus } from '../../../ui/components/StatusCard';
import { ACCESS_TITLE } from '../../../ui/components/ShortcutAccessCard';
import type { CatalogState, ModelAvailability } from '../../../ui/hooks/useModel';
import { websiteAvailability, type WebsiteAvailability } from '../../../ui/hooks/useShortcuts';
import { HOLD_MENU_DESCRIPTION, KEEP_AWAKE_NOTE, LIVE_PREVIEW_DESCRIPTION, SettingsView, SHORTCUT_NOTE } from '../../../ui/popup/SettingsView';
import { PopupView, REMOVED_NOTICE } from '../../../ui/popup/PopupApp';
import { WELCOME_HOLD_HINT, WelcomeView } from '../../../ui/welcome/WelcomeApp';

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
  options: { modelRemoved?: boolean; shortcuts?: WebsiteAvailability; holdMenu?: WebsiteAvailability; catalog?: CatalogState } = {},
) {
  const handlers = {
    onAllowSiteAccess: jest.fn(),
    onTurnOffWebsiteFeatures: jest.fn(),
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
      holdMenu={options.holdMenu ?? 'off'}
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
    expect(screen.getByText('EchoMeBetter · 790 MB · on-device GPU')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
  });

  test('an error can be retried', () => {
    const { onLoadModel } = renderPopup({ state: 'error', message: 'out of memory' });
    expect(screen.getByText('out of memory')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(onLoadModel).toHaveBeenCalled();
  });

  test('every style is listed without a preview badge', () => {
    renderPopup(ready);
    for (const label of ['Professional', 'Grammar', 'Friendly', 'Concise', 'Elaborate']) {
      expect(within(row(label)).queryByText('Preview')).not.toBeInTheDocument();
    }
  });

  test('every style is in one Styles list: installed ones with a check, the others with their size and a download icon', () => {
    const both = catalogWith(['professional', 'grammar']);
    const { onDownload } = renderPopup(ready, installed(['professional'], { state: 'idle' }, both), { catalog: served(both) });
    expect(screen.queryByRole('button', { name: /More styles/ })).not.toBeInTheDocument();
    expect(within(row('Professional')).getByRole('img', { name: 'Professional style installed' })).toBeInTheDocument();
    expect(within(row('Grammar')).getByText('73 MB')).toBeInTheDocument();
    const add = screen.getByRole('button', { name: 'Download the Grammar style (73 MB)' });
    expect(within(row('Grammar')).getByRole('tooltip')).toHaveTextContent('Download GrammarOne-time download · 73 MB. Stored on this device.');
    expect(add).not.toHaveTextContent('Add');
    fireEvent.click(add);
    expect(onDownload).toHaveBeenCalledWith(['grammar']);
    // Styles that run on Professional's download are ready with it.
    for (const label of ['Friendly', 'Concise', 'Elaborate']) {
      expect(within(row(label)).getByRole('img', { name: `${label} style installed` })).toBeInTheDocument();
    }
  });

  test('adapters published after the base was installed are offered from the server', () => {
    renderPopup(ready, installed(['professional']), { catalog: served(catalogWith(['professional', 'grammar'])) });
    expect(screen.getByRole('button', { name: 'Download the Grammar style (73 MB)' })).toBeInTheDocument();
  });

  test('with no style on the device the list says to download one, and the rest say what they wait for', () => {
    const { onDownload } = renderPopup(ready, installed([]));
    expect(screen.getByText('Download a style to start rewriting.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download the Professional style (73 MB)' }));
    expect(onDownload).toHaveBeenCalledWith(['professional']);
    for (const label of ['Grammar', 'Friendly', 'Concise', 'Elaborate']) {
      expect(within(row(label)).getByText('Runs on Professional')).toBeInTheDocument();
      expect(within(row(label)).queryByRole('button')).not.toBeInTheDocument();
    }
  });

  test('a downloaded style shows a check whose tooltip says what that means; no row offers a download', () => {
    renderPopup(ready, installed());
    for (const label of ['Professional', 'Grammar', 'Friendly', 'Concise', 'Elaborate']) {
      expect(within(row(label)).getByRole('img', { name: `${label} style installed` })).toBeInTheDocument();
      expect(within(row(label)).getByRole('tooltip')).toHaveTextContent('InstalledReady to use. Stored on this device, so it works offline.');
    }
    expect(screen.queryByRole('button', { name: /^Download the/ })).not.toBeInTheDocument();
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
    expect(within(row('Professional')).getByRole('img', { name: 'Downloading the Professional style' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /^Download the/ })).not.toBeInTheDocument();
  });
});

describe('keyboard shortcuts in the popup', () => {
  test('the gear opens settings', () => {
    const { onOpenSettings } = renderPopup(ready);
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  test('the style list shows where each style stands, never its keys, whether shortcuts work or not', () => {
    for (const shortcuts of ['on', 'off'] as const) {
      renderPopup(ready, installed(), { shortcuts });
      expect(screen.queryByLabelText(/Shift/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Ctrl\+Shift/)).not.toBeInTheDocument();
      expect(screen.queryByText(ACCESS_TITLE)).not.toBeInTheDocument();
      cleanup();
    }
  });

  test('without site access the popup asks once for what waits on it, and can be told not now', () => {
    const { onAllowSiteAccess, onTurnOffWebsiteFeatures } = renderPopup(ready, installed(), { shortcuts: 'needs-access', holdMenu: 'needs-access' });
    expect(screen.getAllByText(ACCESS_TITLE)).toHaveLength(1);
    expect(screen.getByText(/One-time permission to turn on shortcuts and press and hold\. Chrome will ask you to confirm\. EchoMeBetter only works on text you select, and it never leaves your device\./)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Shift/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    expect(onAllowSiteAccess).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(onTurnOffWebsiteFeatures).toHaveBeenCalledTimes(1);
  });

  test('the press and hold menu alone asks for site access too, and once allowed the card is gone', () => {
    renderPopup(ready, installed(), { holdMenu: 'needs-access' });
    expect(screen.getByText(/One-time permission to turn on press and hold\. Chrome will ask you to confirm/)).toBeInTheDocument();
    cleanup();
    renderPopup(ready, installed(), { holdMenu: 'on' });
    expect(screen.queryByText(ACCESS_TITLE)).not.toBeInTheDocument();
  });
});

function renderSettings(
  shortcuts: WebsiteAvailability,
  options: { platform?: 'mac' | 'other'; installed?: InstalledModelRecord | null; removeReply?: UiReply; focus?: 'keep-awake'; holdMenu?: WebsiteAvailability; livePreview?: boolean | null } = {},
) {
  const handlers = {
    onShortcutsChange: jest.fn(),
    onHoldMenuChange: jest.fn(),
    onLivePreviewChange: jest.fn(),
    onAllowSiteAccess: jest.fn(),
    onKeepLoadedChange: jest.fn(),
    onBack: jest.fn(),
    onRemoveModel: jest.fn(async (): Promise<UiReply> => options.removeReply ?? { ok: true }),
    onRemoveAdapter: jest.fn(async (_adapter: StyleId): Promise<UiReply> => options.removeReply ?? { ok: true }),
  };
  render(<SettingsView platform={options.platform ?? 'other'} shortcuts={shortcuts} holdMenu={options.holdMenu ?? 'on'} livePreview={options.livePreview === undefined ? true : options.livePreview} keepLoaded={15} installed={options.installed ?? null} focus={options.focus} {...handlers} />);
  return handlers;
}

describe('settings page', () => {
  test('the live preview has its own switch, on by default', () => {
    const { onLivePreviewChange } = renderSettings('on');
    const toggle = screen.getByRole('switch', { name: 'Live preview' });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    expect(toggle).toHaveAccessibleDescription(LIVE_PREVIEW_DESCRIPTION);
    fireEvent.click(toggle);
    expect(onLivePreviewChange).toHaveBeenCalledWith(false);
  });

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

  test('the press and hold menu has its own switch', () => {
    const { onHoldMenuChange } = renderSettings('on', { holdMenu: 'on' });
    const toggle = screen.getByRole('switch', { name: 'Press and hold menu' });
    expect(toggle).toHaveAttribute('aria-checked', 'true');
    expect(toggle).toHaveAccessibleDescription(HOLD_MENU_DESCRIPTION);
    fireEvent.click(toggle);
    expect(onHoldMenuChange).toHaveBeenCalledWith(false);
  });

  test('site access is asked for once, under the first setting waiting for it', () => {
    renderSettings('needs-access', { holdMenu: 'needs-access' });
    expect(screen.getAllByRole('button', { name: 'Allow on websites' })).toHaveLength(1);
    cleanup();
    renderSettings('off', { holdMenu: 'needs-access' });
    const mouse = screen.getByRole('region', { name: 'Mouse' });
    expect(within(mouse).getByRole('button', { name: 'Allow on websites' })).toBeInTheDocument();
  });

  test('nothing is switchable before the state is known', () => {
    renderSettings('loading', { holdMenu: 'loading', livePreview: null });
    expect(screen.queryByRole('switch')).not.toBeInTheDocument();
  });

  test('the idle setting reports a number of minutes, and back returns', () => {
    const { onKeepLoadedChange, onBack } = renderSettings('on');
    fireEvent.change(screen.getByLabelText(/Free memory when idle/), { target: { value: '0' } });
    expect(onKeepLoadedChange).toHaveBeenCalledWith(0);
    fireEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(onBack).toHaveBeenCalled();
  });

  test('opened for the keep-awake time, it scrolls to that setting, focuses it and says why', () => {
    const scrolled = jest.fn();
    Element.prototype.scrollIntoView = scrolled;
    renderSettings('on', { focus: 'keep-awake' });
    const select = screen.getByLabelText(/Free memory when idle/);
    expect(select).toHaveFocus();
    expect(scrolled.mock.contexts).toEqual([select]);
    expect(select).toHaveAccessibleDescription(KEEP_AWAKE_NOTE);
  });

  test('opened from the gear, no setting is singled out', () => {
    renderSettings('on');
    expect(screen.getByLabelText(/Free memory when idle/)).not.toHaveFocus();
    expect(screen.queryByText(KEEP_AWAKE_NOTE)).not.toBeInTheDocument();
  });

  test('availability combines the setting with the site access', () => {
    expect(websiteAvailability(null, true)).toBe('loading');
    expect(websiteAvailability(true, null)).toBe('loading');
    expect(websiteAvailability(false, true)).toBe('off');
    expect(websiteAvailability(true, false)).toBe('needs-access');
    expect(websiteAvailability(true, true)).toBe('on');
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
    const both = catalogWith(['professional', 'grammar']);
    const { onRemoveAdapter, onRemoveModel } = renderSettings('off', { installed: record(['professional', 'grammar'], both) });
    fireEvent.click(screen.getByRole('button', { name: 'Remove professional style' }));
    expect(screen.getByText('Remove the Professional style?')).toBeInTheDocument();
    expect(
      screen.getByText("This frees about 73 MB. Professional, Friendly, Concise and Elaborate can't be used until you download it again."),
    ).toBeInTheDocument();
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Remove' })));
    expect(onRemoveAdapter).toHaveBeenCalledWith('professional');
    expect(onRemoveModel).not.toHaveBeenCalled();
  });

  test('the last style cannot be removed alone: the writing model goes with it, and the question says so', async () => {
    const { onRemoveAdapter, onRemoveModel } = renderSettings('off', { installed: record(['professional']) });
    fireEvent.click(screen.getByRole('button', { name: 'Remove professional style' }));
    expect(screen.getByText('Remove the Professional style and the writing model?')).toBeInTheDocument();
    expect(
      screen.getByText(/The Professional style is your only style, and the writing model needs at least one to work, so both are removed\. This frees about 863 MB/),
    ).toBeInTheDocument();
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Remove' })));
    expect(onRemoveModel).toHaveBeenCalledTimes(1);
    expect(onRemoveAdapter).not.toHaveBeenCalled();
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
  test('one download to start: the writing model with the Professional style, and nothing else to choose', () => {
    const { onDownload } = renderPopup({ state: 'unloaded' }, missing());
    const card = screen.getByRole('region', { name: 'Writing model download' });
    expect(within(card).getByText('Set up EchoMeBetter')).toBeInTheDocument();
    expect(within(card).getByText(/Download the writing model and the Professional style once/)).toBeInTheDocument();
    expect(screen.queryByText(MODEL_SOURCE_URL)).not.toBeInTheDocument();
    const included = screen.getByRole('list', { name: 'Included in this download' });
    expect(within(included).getByText('Writing model')).toBeInTheDocument();
    expect(within(included).getByText('Professional style')).toBeInTheDocument();
    expect(screen.getByText('Also powers Grammar, Friendly, Concise and Elaborate')).toBeInTheDocument();
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    // The styles come after the setup: no list, and no download icons, until the model is here.
    expect(screen.queryByText('Styles')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download · 863 MB' }));
    expect(onDownload).toHaveBeenCalledWith(['professional']);
  });

  test('styles with an adapter of their own are named as ones to add later, not downloaded now', () => {
    const more = catalogWith(['professional', 'concise', 'grammar']);
    const { onDownload } = renderPopup({ state: 'unloaded' }, missing(), { catalog: served(more) });
    expect(screen.getByText('Additional styles (Grammar and Concise) are optional and can be downloaded on demand after setup.')).toBeInTheDocument();
    expect(screen.getByText('Also powers Friendly and Elaborate')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download · 863 MB' }));
    expect(onDownload).toHaveBeenCalledWith(['professional']);
  });

  test('while the starter download runs, the styles stay out of the way', () => {
    renderPopup({ state: 'unloaded' }, missing({ state: 'downloading', target: FIRST, phase: 'fetching', receivedBytes: 1, totalBytes: 10, bytesPerSecond: 0 }));
    expect(screen.queryByText('Styles')).not.toBeInTheDocument();
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
    shortcuts: 'off' as WebsiteAvailability,
    holdMenu: 'off' as WebsiteAvailability,
    onAllowSiteAccess: jest.fn(),
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

  test('the styles to download sit in the setup area, as in the popup, and only once the model is there', () => {
    const both = catalogWith(['professional', 'grammar']);
    render(<WelcomeView {...props} catalog={served(both)} model={installed(['professional'], { state: 'idle' }, both)} />);
    const tryIt = screen.getByRole('region', { name: 'Try it here' });
    const list = within(tryIt).getByRole('region', { name: 'Styles' });
    expect(within(list).getByRole('img', { name: 'Professional style installed' })).toBeInTheDocument();
    expect(within(list).getByText('73 MB')).toBeInTheDocument();
    fireEvent.click(within(list).getByRole('button', { name: 'Download the Grammar style (73 MB)' }));
    expect(props.onDownload).toHaveBeenCalledWith(['grammar']);
    cleanup();
    render(<WelcomeView {...props} model={missing()} />);
    expect(screen.queryByRole('region', { name: 'Styles' })).not.toBeInTheDocument();
  });

  test('the Five writing styles section only describes the styles: no download, whatever is installed', () => {
    const both = catalogWith(['professional', 'grammar']);
    render(<WelcomeView {...props} catalog={served(both)} model={installed(['professional'], { state: 'idle' }, both)} />);
    const about = screen.getByRole('region', { name: 'Five writing styles' });
    expect(within(about).queryByRole('button')).not.toBeInTheDocument();
    expect(within(about).queryByText(/MB/)).not.toBeInTheDocument();
    expect(within(about).getByText('Fix grammar and spelling only')).toBeInTheDocument();
  });

  test('teaches the shortcuts while they are on, and asks for site access when needed', () => {
    render(<WelcomeView {...props} model={installed()} shortcuts="needs-access" />);
    expect(screen.getByText(/press Alt\+Shift\+P for Professional/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    expect(props.onAllowSiteAccess).toHaveBeenCalled();
  });

  test('teaches the press and hold while it is on', () => {
    render(<WelcomeView {...props} model={installed()} holdMenu="on" />);
    expect(screen.getByText(WELCOME_HOLD_HINT)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Allow on websites' })).not.toBeInTheDocument();
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
