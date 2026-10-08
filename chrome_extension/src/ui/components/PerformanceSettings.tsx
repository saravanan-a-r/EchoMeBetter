/**
 * Where the model runs and how much of the computer it may use; the same on
 * the popup's settings page and the welcome page.
 *
 * The GPU/CPU choice appears only where the browser offers a GPU, and the
 * GPU is the default there. The processor's share is offered whenever
 * rewrites run on it; the GPU's power setting whenever they run on the GPU.
 * A GPU that couldn't run the model is said so plainly, with a way to try
 * it again (after a browser or driver update, say).
 */
import type { ChangeEvent } from 'react';
import { CPU_USAGES, GPU_POWERS, threadsFor, type ComputeSettings, type CpuUsage, type GpuPower, type GpuProblem, type Processor } from '../../shared/compute';
import type { EngineStatus } from '../../shared/status';
import type { GpuAvailability } from '../hooks/useCompute';
import { SECONDARY_BUTTON } from './buttons';

export interface PerformanceSettingsProps {
  readonly compute: ComputeSettings;
  readonly gpu: GpuAvailability;
  readonly gpuProblem: GpuProblem | null;
  /** `navigator.hardwareConcurrency` and `crossOriginIsolated`: what the thread counts are worked out from. */
  readonly cores: number;
  readonly crossOriginIsolated: boolean;
  readonly status: EngineStatus;
  readonly onChange: (next: ComputeSettings) => void;
  readonly onRetryGpu: () => void;
}

const USAGE_LABELS: Record<CpuUsage, string> = { light: 'Light', balanced: 'Balanced', maximum: 'Maximum' };
const POWER_LABELS: Record<GpuPower, string> = { 'high-performance': 'High performance', 'low-power': 'Power saving' };
const VENDOR_NAMES: Record<string, string> = { amd: 'AMD', apple: 'Apple', arm: 'Arm', intel: 'Intel', nvidia: 'NVIDIA', qualcomm: 'Qualcomm' };

export const GPU_PROBLEM_NOTE = "Your graphics chip couldn't run the model, so rewrites run on the processor.";

function vendorName(vendor: string): string {
  return VENDOR_NAMES[vendor] ?? (vendor ? vendor.charAt(0).toUpperCase() + vendor.slice(1) : '');
}

function cores(count: number): string {
  return `${count} ${count === 1 ? 'core' : 'cores'}`;
}

const SELECT =
  '_echo_$_rounded-lg _echo_$_border _echo_$_border-line-strong _echo_$_bg-surface-raised _echo_$_px-2 _echo_$_py-1 _echo_$_text-sm';
const HINT = '_echo_$_mt-1 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-muted';

function SelectRow<T extends string>(props: {
  id: string;
  label: string;
  hint: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (value: T) => void;
}) {
  return (
    <div>
      <div className="_echo_$_flex _echo_$_items-center _echo_$_justify-between _echo_$_gap-3">
        <label htmlFor={props.id} className="_echo_$_text-sm _echo_$_font-medium">
          {props.label}
        </label>
        <select
          id={props.id}
          value={props.value}
          aria-describedby={`${props.id}-hint`}
          onChange={(event: ChangeEvent<HTMLSelectElement>) => props.onChange(event.target.value as T)}
          className={SELECT}
        >
          {props.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
      <p id={`${props.id}-hint`} className={HINT}>
        {props.hint}
      </p>
    </div>
  );
}

function ProcessorChoice({ value, onChange }: { value: Processor; onChange: (processor: Processor) => void }) {
  const choices: { value: Processor; title: string; subtitle: string }[] = [
    { value: 'gpu', title: 'GPU', subtitle: 'Graphics chip · fastest' },
    { value: 'cpu', title: 'CPU', subtitle: 'Processor' },
  ];
  return (
    <div role="radiogroup" aria-labelledby="processor-label" className="_echo_$_mt-2 _echo_$_grid _echo_$_grid-cols-2 _echo_$_gap-1 _echo_$_rounded-xl _echo_$_bg-muted _echo_$_p-1">
      {choices.map((choice) => {
        const checked = choice.value === value;
        return (
          <label
            key={choice.value}
            className={`_echo_$_cursor-pointer _echo_$_rounded-lg _echo_$_px-3 _echo_$_py-1.5 _echo_$_transition has-[:focus-visible]:_echo_$_outline has-[:focus-visible]:_echo_$_outline-2 has-[:focus-visible]:_echo_$_outline-focus ${
              checked
                ? '_echo_$_bg-segment-active _echo_$_shadow-sm'
                : '_echo_$_text-fg-soft hover:_echo_$_bg-segment-hover'
            }`}
          >
            <input
              type="radio"
              name="processor"
              value={choice.value}
              checked={checked}
              onChange={() => onChange(choice.value)}
              aria-describedby={`processor-${choice.value}-subtitle`}
              className="_echo_$_sr-only"
            />
            <span className={`_echo_$_block _echo_$_text-sm _echo_$_font-semibold ${checked ? '_echo_$_text-accent-fg-strong' : ''}`}>{choice.title}</span>
            <span id={`processor-${choice.value}-subtitle`} className="_echo_$_block _echo_$_text-[11px] _echo_$_text-fg-muted">
              {choice.subtitle}
            </span>
          </label>
        );
      })}
    </div>
  );
}

function runningNow(status: EngineStatus): string | null {
  if (status.state === 'loading') return 'Loading the model…';
  if (status.state !== 'ready') return null;
  return status.runningOn.processor === 'gpu'
    ? 'Running on the GPU now.'
    : `Running on the processor now, with ${status.runningOn.threads} ${status.runningOn.threads === 1 ? 'thread' : 'threads'}.`;
}

export function PerformanceSettings(props: PerformanceSettingsProps) {
  const { compute, gpu, gpuProblem, onChange } = props;
  if (gpu.state === 'checking') return null;
  const offered = gpu.state === 'available';
  const onGpu = offered && compute.processor === 'gpu' && !gpuProblem;
  const vendor = offered ? vendorName(gpu.gpu.vendor) : '';
  const now = runningNow(props.status);
  const usageOptions = CPU_USAGES.map((usage) => ({
    value: usage,
    label: `${USAGE_LABELS[usage]} · ${cores(threadsFor(usage, props.cores, props.crossOriginIsolated))}`,
  }));

  return (
    <div className="_echo_$_space-y-3">
      {offered ? (
        <div>
          <p id="processor-label" className="_echo_$_text-sm _echo_$_font-medium">
            Run the model on
          </p>
          <ProcessorChoice value={compute.processor} onChange={(processor) => onChange({ ...compute, processor })} />
          {compute.processor === 'gpu' && gpuProblem ? (
            <div role="status" className="_echo_$_mt-2 _echo_$_flex _echo_$_items-start _echo_$_justify-between _echo_$_gap-3 _echo_$_rounded-lg _echo_$_bg-danger-subtle _echo_$_p-2.5">
              <p className="_echo_$_text-xs _echo_$_leading-5 _echo_$_text-fg-secondary">{GPU_PROBLEM_NOTE}</p>
              <button type="button" onClick={props.onRetryGpu} className={SECONDARY_BUTTON}>
                Try the GPU again
              </button>
            </div>
          ) : (
            <p className={HINT}>
              {compute.processor === 'gpu'
                ? `Rewrites run on your ${vendor ? `${vendor} ` : ''}graphics chip: the fastest way, and the processor stays free for your other apps.`
                : 'Rewrites run on the processor. The GPU is usually faster.'}
            </p>
          )}
        </div>
      ) : null}

      {onGpu ? (
        <SelectRow
          id="gpu-power"
          label="GPU power"
          hint="Power saving uses the built-in graphics on computers that have two, which is easier on the battery but slower."
          value={compute.gpuPower}
          options={GPU_POWERS.map((power) => ({ value: power, label: POWER_LABELS[power] }))}
          onChange={(gpuPower) => onChange({ ...compute, gpuPower })}
        />
      ) : (
        <SelectRow
          id="cpu-usage"
          label="Processor use"
          hint={`How much of the processor (${cores(props.cores)}) a rewrite may use. More is faster, but other apps can slow down while it runs.`}
          value={compute.cpuUsage}
          options={usageOptions}
          onChange={(cpuUsage) => onChange({ ...compute, cpuUsage })}
        />
      )}

      {now ? (
        <p aria-live="polite" className="_echo_$_flex _echo_$_items-center _echo_$_gap-1.5 _echo_$_border-t _echo_$_border-line _echo_$_pt-2.5 _echo_$_text-xs _echo_$_text-fg-muted">
          <span aria-hidden="true" className={`_echo_$_h-1.5 _echo_$_w-1.5 _echo_$_rounded-full ${props.status.state === 'ready' ? '_echo_$_bg-success' : '_echo_$_bg-info _echo_$_animate-pulse'}`} />
          {now}
        </p>
      ) : null}
    </div>
  );
}
