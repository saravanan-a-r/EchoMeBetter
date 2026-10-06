/**
 * What is downloaded, and removing it to free disk space. A style can be
 * removed on its own; removing the writing model removes every style with
 * it. Each removal asks first, in place, and says plainly what it means.
 */
import { useState } from 'react';
import { formatBytes, formatList } from '../../shared/format';
import type { UiReply } from '../../shared/messages';
import { stylesUsing, type InstalledModelRecord } from '../../shared/modelInstall';
import { STYLE_IDS, styleLabel, type StyleId } from '../../shared/styles';
import { DANGER_BUTTON, SECONDARY_BUTTON } from './buttons';

export interface StorageSettingsProps {
  readonly installed: InstalledModelRecord | null;
  readonly onRemoveModel: () => Promise<UiReply>;
  readonly onRemoveAdapter: (adapter: StyleId) => Promise<UiReply>;
}

interface Item {
  readonly key: 'model' | StyleId;
  readonly label: string;
  readonly sizeBytes: number;
  readonly question: string;
  readonly consequence: string;
  readonly remove: () => Promise<UiReply>;
}

function items(installed: InstalledModelRecord, props: StorageSettingsProps): Item[] {
  const { catalog } = installed;
  const adapters = STYLE_IDS.flatMap((style) => (installed.adapters.includes(style) ? catalog.adapters.filter((adapter) => adapter.style === style) : []));
  const total = catalog.model.sizeBytes + adapters.reduce((sum, adapter) => sum + adapter.sizeBytes, 0);
  return [
    {
      key: 'model',
      label: 'Writing model',
      sizeBytes: catalog.model.sizeBytes,
      question: 'Remove the writing model and every style?',
      consequence: `This frees about ${formatBytes(total)} of disk space. EchoMeBetter can't rewrite text until you download it again. Your settings are kept.`,
      remove: props.onRemoveModel,
    },
    ...adapters.map(
      (adapter): Item => ({
        key: adapter.style,
        label: `${styleLabel(adapter.style)} style`,
        sizeBytes: adapter.sizeBytes,
        question: `Remove the ${styleLabel(adapter.style)} style?`,
        consequence: `This frees about ${formatBytes(adapter.sizeBytes)}. ${formatList(stylesUsing(catalog, adapter.style).map(styleLabel))} can't be used until you download it again.`,
        remove: () => props.onRemoveAdapter(adapter.style),
      }),
    ),
  ];
}

function Confirm({ item, onDone }: { item: Item; onDone: () => void }) {
  const [removing, setRemoving] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const remove = async () => {
    setRemoving(true);
    setProblem(null);
    const reply = await item.remove().catch((): UiReply => ({ ok: false, error: { code: 'INTERNAL' } }));
    setRemoving(false);
    if (reply.ok) return onDone();
    setProblem(reply.error.code === 'BUSY' ? 'Wait for the current rewrite to finish.' : "Couldn't remove it. Please try again.");
  };
  return (
    <div role="group" aria-labelledby={`remove-${item.key}-title`} className="_echo_$_mt-2 _echo_$_rounded-xl _echo_$_border _echo_$_border-danger-400/40 _echo_$_bg-danger-50 _echo_$_p-3 dark:_echo_$_bg-danger-950">
      <p id={`remove-${item.key}-title`} className="_echo_$_text-sm _echo_$_font-semibold _echo_$_text-ink-900 dark:_echo_$_text-white">
        {item.question}
      </p>
      <p className="_echo_$_mt-1 _echo_$_text-xs _echo_$_leading-5 _echo_$_text-ink-600 dark:_echo_$_text-ink-200">{item.consequence}</p>
      {problem ? (
        <p role="alert" className="_echo_$_mt-2 _echo_$_text-xs _echo_$_font-medium _echo_$_text-danger-500">
          {problem}
        </p>
      ) : null}
      <div className="_echo_$_mt-2.5 _echo_$_flex _echo_$_justify-end _echo_$_gap-2">
        <button type="button" disabled={removing} onClick={onDone} className={SECONDARY_BUTTON}>
          Keep it
        </button>
        <button type="button" disabled={removing} onClick={() => void remove()} className={DANGER_BUTTON}>
          {removing ? 'Removing…' : 'Remove'}
        </button>
      </div>
    </div>
  );
}

export function StorageSettings(props: StorageSettingsProps) {
  const [confirming, setConfirming] = useState<Item['key'] | null>(null);
  if (!props.installed) {
    return <p className="_echo_$_text-sm _echo_$_text-ink-500 dark:_echo_$_text-ink-300">Nothing downloaded yet.</p>;
  }
  return (
    <ul aria-label="Downloaded" className="_echo_$_space-y-2">
      {items(props.installed, props).map((item) => (
        <li key={item.key}>
          <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-3 _echo_$_text-sm">
            <span className="_echo_$_font-medium">{item.label}</span>
            <span className="_echo_$_ml-auto _echo_$_text-xs _echo_$_tabular-nums _echo_$_text-ink-500 dark:_echo_$_text-ink-300">{formatBytes(item.sizeBytes)}</span>
            <button
              type="button"
              aria-label={`Remove ${item.label.toLowerCase()}`}
              onClick={() => setConfirming(item.key)}
              className="_echo_$_rounded _echo_$_text-xs _echo_$_font-medium _echo_$_text-ink-500 _echo_$_underline-offset-2 hover:_echo_$_text-danger-500 hover:_echo_$_underline focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-echo-500 dark:_echo_$_text-ink-300"
            >
              Remove
            </button>
          </div>
          {confirming === item.key ? <Confirm item={item} onDone={() => setConfirming(null)} /> : null}
        </li>
      ))}
    </ul>
  );
}
