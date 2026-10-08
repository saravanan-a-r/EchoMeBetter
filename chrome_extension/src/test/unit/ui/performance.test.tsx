/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { DEFAULT_COMPUTE, type ComputeSettings } from '../../../shared/compute';
import { MODEL_SOURCE_URL } from '../../../shared/modelSource';
import type { InstalledModelRecord } from '../../../shared/modelInstall';
import type { EngineStatus, ModelSummary } from '../../../shared/status';
import { GPU_PROBLEM_NOTE, PerformanceSettings, type PerformanceSettingsProps } from '../../../ui/components/PerformanceSettings';
import { SettingsView } from '../../../ui/popup/SettingsView';
import { WelcomeView } from '../../../ui/welcome/WelcomeApp';

const summary: ModelSummary = { id: 'echomebetter-int8', displayName: 'EchoMeBetter', precision: 'int8', sizeBytes: 790_000_000 };
const ready: EngineStatus = { state: 'ready', model: summary, runningOn: { processor: 'gpu', threads: 4 } };

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

function renderSettings(performanceProps: PerformanceSettingsProps | undefined) {
  render(
    <SettingsView
      platform="other"
      shortcuts="off"
      keepLoaded={15}
      installed={null}
      performance={performanceProps}
      onShortcutsChange={jest.fn()}
      onAllowSiteAccess={jest.fn()}
      onKeepLoadedChange={jest.fn()}
      onBack={jest.fn()}
      onRemoveModel={jest.fn(async () => ({ ok: true }) as const)}
      onRemoveAdapter={jest.fn(async () => ({ ok: true }) as const)}
    />,
  );
}

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
    renderSettings(performance());
    expect(within(screen.getByRole('region', { name: 'Performance' })).getByRole('radio', { name: /GPU/ })).toBeChecked();
  });

  test('the settings page leaves it out while the settings are being read', () => {
    renderSettings(undefined);
    expect(screen.queryByRole('region', { name: 'Performance' })).not.toBeInTheDocument();
  });
});

describe('welcome page', () => {
  const installed: InstalledModelRecord = {
    sourceUrl: MODEL_SOURCE_URL,
    catalog: { base: 'base-1', model: summary, adapters: [{ style: 'professional', sizeBytes: 73_000_000 }], fallbackAdapter: 'professional' },
    adapters: ['professional'],
  };
  const welcome = {
    status: { state: 'unloaded' } as EngineStatus,
    model: { known: true, installed, download: { state: 'idle' } } as const,
    catalog: { status: 'loading' } as const,
    onLoadModel: jest.fn(),
    onDownload: jest.fn(),
    onCancelDownload: jest.fn(),
    onRetryCatalog: jest.fn(),
    platform: 'other' as const,
    shortcuts: 'off' as const,
    onAllowSiteAccess: jest.fn(),
  };

  test('lets the user choose GPU or CPU where a GPU is on offer, saying where to change it later', () => {
    render(<WelcomeView {...welcome} performance={performance()} />);
    const section = screen.getByRole('region', { name: 'Speed and power' });
    expect(within(section).getByText(/runs on your graphics chip \(GPU\)/)).toBeInTheDocument();
    expect(within(section).getByText(/settings of the toolbar popup/)).toBeInTheDocument();
    expect(within(section).getByRole('radio', { name: /GPU/ })).toBeChecked();
  });

  test('without a GPU it speaks only of the processor', () => {
    render(<WelcomeView {...welcome} performance={performance({ gpu: { state: 'none' } })} />);
    const section = screen.getByRole('region', { name: 'Speed and power' });
    expect(within(section).queryByText(/GPU|graphics/)).not.toBeInTheDocument();
    expect(within(section).getByLabelText('Processor use')).toBeInTheDocument();
  });
});
