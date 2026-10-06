/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { formatBytes, formatDuration, formatTimeLeft } from '../../../shared/format';
import type { UiReply } from '../../../shared/messages';
import type { DownloadState } from '../../../shared/modelInstall';
import { MODEL_SOURCE_URL } from '../../../shared/modelSource';
import type { EngineStatus, ModelSummary } from '../../../shared/status';
import { presentDownload } from '../../../ui/components/ModelDownloadCard';
import { presentStatus } from '../../../ui/components/StatusCard';
import { ACCESS_TITLE } from '../../../ui/components/ShortcutAccessCard';
import type { ModelAvailability } from '../../../ui/hooks/useModel';
import { shortcutAvailability, type ShortcutAvailability } from '../../../ui/hooks/useShortcuts';
import { SettingsView, SHORTCUT_NOTE } from '../../../ui/popup/SettingsView';
import { PopupView, REMOVED_NOTICE } from '../../../ui/popup/PopupApp';
import { WelcomeView } from '../../../ui/welcome/WelcomeApp';

const summary: ModelSummary = { id: 'flan', displayName: 'Flan-T5 Large', placeholder: true, precision: 'int8', sizeBytes: 820_000_000 };
const installed: ModelAvailability = { known: true, installed: { sourceUrl: MODEL_SOURCE_URL, model: summary }, download: { state: 'idle' } };
const missing = (download: DownloadState = { state: 'idle' }): ModelAvailability => ({ known: true, installed: null, download });

function renderPopup(
  status: EngineStatus,
  model: ModelAvailability = installed,
  options: { modelRemoved?: boolean; removeReply?: UiReply; shortcuts?: ShortcutAvailability } = {},
) {
  const handlers = {
    onAllowSiteAccess: jest.fn(),
    onTurnOffShortcuts: jest.fn(),
    onOpenSettings: jest.fn(),
    onLoadModel: jest.fn(),
    onOpenGuide: jest.fn(),
    onDownloadModel: jest.fn(),
    onCancelDownload: jest.fn(),
    onRemoveModel: jest.fn(async (): Promise<UiReply> => options.removeReply ?? { ok: true }),
  };
  render(<PopupView status={status} model={model} modelRemoved={options.modelRemoved ?? false} platform="mac" shortcuts={options.shortcuts ?? 'off'} {...handlers} />);
  return handlers;
}

const ready: EngineStatus = { state: 'ready', model: summary };

describe('popup with the model downloaded', () => {
  test('sleeping model offers to load now', () => {
    const { onLoadModel } = renderPopup({ state: 'unloaded' });
    fireEvent.click(screen.getByRole('button', { name: 'Load now' }));
    expect(onLoadModel).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('button', { name: 'Download model' })).not.toBeInTheDocument();
  });

  test('loading shows progress', () => {
    renderPopup({ state: 'loading', progress: 0.37 });
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '37');
    expect(screen.getByText('Loading · 37%')).toBeInTheDocument();
  });

  test('ready names the model and flags a stand-in model', () => {
    renderPopup(ready);
    expect(screen.getByText(/Flan-T5 Large · int8 · 820 MB/)).toBeInTheDocument();
    expect(screen.getByText('Preview model')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
  });

  test('an error can be retried', () => {
    const { onLoadModel } = renderPopup({ state: 'error', message: 'out of memory' });
    expect(screen.getByText('out of memory')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(onLoadModel).toHaveBeenCalled();
  });

  test('every style is listed', () => {
    renderPopup(ready);
    for (const label of ['Professional', 'Grammar', 'Friendly', 'Concise', 'Elaborate']) expect(screen.getByText(label)).toBeInTheDocument();
  });
});

describe('keyboard shortcuts in the popup', () => {
  test('the gear opens settings', () => {
    const { onOpenSettings } = renderPopup(ready);
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    expect(onOpenSettings).toHaveBeenCalledTimes(1);
  });

  test('working shortcuts are listed next to their styles, for this platform', () => {
    renderPopup(ready, installed, { shortcuts: 'on' });
    expect(screen.getByLabelText('Control Shift P')).toHaveTextContent('⌃⇧P');
    expect(screen.queryByText(ACCESS_TITLE)).not.toBeInTheDocument();
  });

  test('turned off, no keys are shown and nothing is asked', () => {
    renderPopup(ready, installed, { shortcuts: 'off' });
    expect(screen.queryByLabelText(/Shift/)).not.toBeInTheDocument();
    expect(screen.getByText('Polished and workplace-ready')).toBeInTheDocument();
    expect(screen.queryByText(ACCESS_TITLE)).not.toBeInTheDocument();
  });

  test('without site access the popup asks, and can be told not now', () => {
    const { onAllowSiteAccess, onTurnOffShortcuts } = renderPopup(ready, installed, { shortcuts: 'needs-access' });
    expect(screen.getByText(ACCESS_TITLE)).toBeInTheDocument();
    expect(screen.getByText(/shortcuts like ⌃⇧P on websites/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Shift/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    expect(onAllowSiteAccess).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Not now' }));
    expect(onTurnOffShortcuts).toHaveBeenCalledTimes(1);
  });
});

describe('settings page', () => {
  function renderSettings(shortcuts: ShortcutAvailability, platform: 'mac' | 'other' = 'other') {
    const handlers = { onShortcutsChange: jest.fn(), onAllowSiteAccess: jest.fn(), onKeepLoadedChange: jest.fn(), onBack: jest.fn() };
    render(<SettingsView platform={platform} shortcuts={shortcuts} keepLoaded={15} {...handlers} />);
    return handlers;
  }

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
    const { onAllowSiteAccess } = renderSettings('needs-access', 'mac');
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

describe('popup without the model', () => {
  test('offers the download, and shows where it comes from', () => {
    const { onDownloadModel } = renderPopup({ state: 'unloaded' }, missing());
    expect(screen.getByText('Writing model needed')).toBeInTheDocument();
    expect(screen.getByText(MODEL_SOURCE_URL)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Remove downloaded model' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download model' }));
    expect(onDownloadModel).toHaveBeenCalledTimes(1);
  });

  test('shows the download progress, speed and time left', () => {
    renderPopup(
      { state: 'unloaded' },
      missing({ state: 'downloading', phase: 'fetching', receivedBytes: 312_000_000, totalBytes: 820_000_000, bytesPerSecond: 24_000_000 }),
    );
    expect(screen.getByText('Downloading · 38%')).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Download progress' })).toHaveAttribute('aria-valuenow', '38');
    expect(screen.getByText('312 MB of 820 MB · 24 MB/s · about 21 s left')).toBeInTheDocument();
    expect(screen.getByText('You can close this window. The download keeps going.')).toBeInTheDocument();
  });

  test('cancelling asks first, and can be backed out of', () => {
    const { onCancelDownload } = renderPopup(
      { state: 'unloaded' },
      missing({ state: 'downloading', phase: 'fetching', receivedBytes: 1, totalBytes: 10, bytesPerSecond: 0 }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Cancel download' }));
    expect(screen.getByText('Stop and delete the partly downloaded files?')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Keep downloading' }));
    expect(onCancelDownload).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Cancel download' }));
    fireEvent.click(screen.getByRole('button', { name: 'Stop download' }));
    expect(onCancelDownload).toHaveBeenCalledTimes(1);
  });

  test('a stopped download explains why and can be tried again', () => {
    const { onDownloadModel } = renderPopup({ state: 'unloaded' }, missing({ state: 'failed', error: { code: 'STORAGE_FULL', details: { neededBytes: 300_000_000 } } }));
    expect(screen.getByText('Download stopped')).toBeInTheDocument();
    expect(screen.getByText(/Free up 300 MB and try again/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(onDownloadModel).toHaveBeenCalledTimes(1);
  });

  test('after removing the model the popup says so', () => {
    renderPopup({ state: 'unloaded' }, missing(), { modelRemoved: true });
    expect(screen.getByText(REMOVED_NOTICE)).toBeInTheDocument();
  });

  test('nothing model-related renders before storage has been read', () => {
    renderPopup({ state: 'unloaded' }, { known: false, installed: null, download: { state: 'idle' } });
    expect(screen.queryByRole('button', { name: 'Download model' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Load now' })).not.toBeInTheDocument();
  });
});

describe('removing the model', () => {
  test('asks first, says what it means, and can be kept', () => {
    const { onRemoveModel } = renderPopup(ready);
    fireEvent.click(screen.getByRole('button', { name: 'Remove downloaded model' }));
    expect(screen.getByText('Remove the model from this device?')).toBeInTheDocument();
    expect(
      screen.getByText("This frees about 820 MB of disk space. EchoMeBetter can't rewrite text until you download the model again. Your settings are kept."),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Keep it' }));
    expect(onRemoveModel).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Remove downloaded model' })).toBeInTheDocument();
  });

  test('confirming removes it', async () => {
    const { onRemoveModel } = renderPopup(ready);
    fireEvent.click(screen.getByRole('button', { name: 'Remove downloaded model' }));
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Remove model' })));
    expect(onRemoveModel).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Removing…' })).toBeDisabled();
  });

  test('a rewrite in progress blocks it, with a reason', async () => {
    renderPopup(ready, installed, { removeReply: { ok: false, error: { code: 'BUSY' } } });
    fireEvent.click(screen.getByRole('button', { name: 'Remove downloaded model' }));
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Remove model' })));
    expect(screen.getByRole('alert')).toHaveTextContent('Wait for the current rewrite to finish.');
    expect(screen.getByRole('button', { name: 'Remove model' })).toBeEnabled();
  });
});

describe('welcome page', () => {
  const props = {
    status: { state: 'unloaded' } as EngineStatus,
    onLoadModel: jest.fn(),
    onDownloadModel: jest.fn(),
    onCancelDownload: jest.fn(),
    platform: 'other' as const,
    shortcuts: 'off' as ShortcutAvailability,
    onAllowSiteAccess: jest.fn(),
  };

  test('offers the download next to the practice box until the model is there', () => {
    render(<WelcomeView {...props} model={missing()} />);
    expect(screen.getByText(/Download the model, then select text below/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Download model' }));
    expect(props.onDownloadModel).toHaveBeenCalled();
  });

  test('shows the model status once downloaded', () => {
    render(<WelcomeView {...props} model={installed} />);
    expect(screen.getByRole('button', { name: 'Load now' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Download model' })).not.toBeInTheDocument();
  });

  test('teaches the shortcuts while they are on, and asks for site access when needed', () => {
    render(<WelcomeView {...props} model={installed} shortcuts="needs-access" />);
    expect(screen.getByText(/press Alt\+Shift\+P for Professional/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Allow on websites' }));
    expect(props.onAllowSiteAccess).toHaveBeenCalled();
  });

  test('with shortcuts off it only teaches the right-click', () => {
    render(<WelcomeView {...props} model={installed} />);
    expect(screen.queryByText(/press Alt\+Shift/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Allow on websites' })).not.toBeInTheDocument();
  });
});

test('status and download presentation', () => {
  expect(presentStatus({ state: 'unloaded' }).action).toBe('Load now');
  expect(presentDownload({ state: 'downloading', phase: 'fetching', receivedBytes: 0, totalBytes: 0, bytesPerSecond: 0 })).toEqual({
    label: 'Downloading',
    detail: 'Connecting…',
    percent: 0,
  });
  expect(presentDownload({ state: 'downloading', phase: 'verifying', receivedBytes: 9, totalBytes: 9, bytesPerSecond: 0 }).label).toBe('Checking files…');
});

test('sizes and times', () => {
  expect(formatBytes(1_500_000_000)).toBe('1.5 GB');
  expect(formatBytes(820_000_000)).toBe('820 MB');
  expect(formatBytes(512_000)).toBe('512 KB');
  expect(formatTimeLeft(21.4)).toBe('about 21 s left');
  expect(formatTimeLeft(150)).toBe('about 3 min left');
  expect(formatTimeLeft(3900)).toBe('about 1 h 5 min left');
});

test('rewrite durations', () => {
  expect(formatDuration(420)).toBe('420 ms');
  expect(formatDuration(999.4)).toBe('999 ms');
  expect(formatDuration(1000)).toBe('1.0 s');
  expect(formatDuration(2345)).toBe('2.3 s');
  expect(formatDuration(60_000)).toBe('1 min');
  expect(formatDuration(95_000)).toBe('1 min 35 s');
});
