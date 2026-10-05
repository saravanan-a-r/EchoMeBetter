/**
 * Every message that crosses a process boundary, in one place.
 *
 *   page (foreground) ──port "echo/job"──▶ service worker ──runtime──▶ offscreen document ──postMessage──▶ worker
 *
 * The service worker is the hub: it owns the context menu, injects the
 * foreground script, keeps the offscreen document alive, and routes each
 * job's events back to the port of the frame that asked for it. Nothing on
 * the page ever talks to the model directly.
 */
import type { ErrorPayload } from './errors';
import type { DownloadState, InstalledModelRecord } from './modelInstall';
import { isStyleId, type StyleId } from './styles';
import type { EngineStatus } from './status';

export const JOB_PORT_NAME = 'echo/job';

// ---------------------------------------------------------------------------
// service worker → foreground (chrome.tabs.sendMessage, per frame)
// ---------------------------------------------------------------------------

export interface StartJobMessage {
  readonly kind: 'echo/start-job';
  readonly jobId: string;
  readonly style: StyleId;
}

/** Shown in the top frame when the frame that was clicked can't be reached. */
export interface NoticeMessage {
  readonly kind: 'echo/notice';
  readonly error: ErrorPayload;
}

export type ForegroundMessage = StartJobMessage | NoticeMessage;

export type StartJobReply = { readonly ok: true } | { readonly ok: false; readonly error: ErrorPayload };

export function isForegroundMessage(value: unknown): value is ForegroundMessage {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as { kind?: unknown; jobId?: unknown; style?: unknown; error?: unknown };
  if (message.kind === 'echo/start-job') return typeof message.jobId === 'string' && isStyleId(message.style);
  if (message.kind === 'echo/notice') return typeof message.error === 'object' && message.error !== null;
  return false;
}

// ---------------------------------------------------------------------------
// foreground ⇄ service worker, over a Port named JOB_PORT_NAME (one per job)
// ---------------------------------------------------------------------------

export type JobRequest =
  | { readonly kind: 'job/request'; readonly jobId: string; readonly style: StyleId; readonly text: string }
  | { readonly kind: 'job/cancel'; readonly jobId: string };

export type JobPhase = 'loading-model' | 'rewriting';

export type JobEvent =
  | { readonly kind: 'job/phase'; readonly jobId: string; readonly phase: JobPhase; readonly progress?: number }
  | { readonly kind: 'job/done'; readonly jobId: string; readonly text: string }
  | { readonly kind: 'job/failed'; readonly jobId: string; readonly error: ErrorPayload };

export function isJobRequest(value: unknown): value is JobRequest {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as { kind?: unknown; jobId?: unknown; style?: unknown; text?: unknown };
  if (typeof message.jobId !== 'string') return false;
  if (message.kind === 'job/cancel') return true;
  return message.kind === 'job/request' && isStyleId(message.style) && typeof message.text === 'string';
}

export function isJobEvent(value: unknown): value is JobEvent {
  if (typeof value !== 'object' || value === null) return false;
  const kind = (value as { kind?: unknown }).kind;
  return kind === 'job/phase' || kind === 'job/done' || kind === 'job/failed';
}

// ---------------------------------------------------------------------------
// service worker ⇄ offscreen document (chrome.runtime.sendMessage)
// ---------------------------------------------------------------------------

export type OffscreenCommand =
  | { readonly target: 'offscreen'; readonly kind: 'engine/ping' }
  | { readonly target: 'offscreen'; readonly kind: 'engine/warm-up' }
  | { readonly target: 'offscreen'; readonly kind: 'engine/rewrite'; readonly jobId: string; readonly style: StyleId; readonly text: string }
  | { readonly target: 'offscreen'; readonly kind: 'engine/cancel'; readonly jobId: string }
  | { readonly target: 'offscreen'; readonly kind: 'model/download' }
  | { readonly target: 'offscreen'; readonly kind: 'model/cancel-download' }
  | { readonly target: 'offscreen'; readonly kind: 'model/remove' };

export interface OffscreenEventMessage {
  readonly target: 'background';
  readonly kind: 'engine/event';
  readonly event: EngineEvent;
}

export function isOffscreenCommand(value: unknown): value is OffscreenCommand {
  return typeof value === 'object' && value !== null && (value as { target?: unknown }).target === 'offscreen';
}

export function isOffscreenEventMessage(value: unknown): value is OffscreenEventMessage {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as { target?: unknown }).target === 'background' &&
    (value as { kind?: unknown }).kind === 'engine/event'
  );
}

// ---------------------------------------------------------------------------
// offscreen document ⇄ inference worker (postMessage)
// ---------------------------------------------------------------------------

export interface WorkerConfig {
  /** Absolute URL of the folder the model is downloaded from (model.json and the files it names). */
  readonly modelSourceUrl: string;
  /** Absolute URL of the folder holding onnxruntime-web's .wasm binary. */
  readonly wasmBaseUrl: string;
  readonly threads: number;
}

export type WorkerRequest =
  | { readonly type: 'configure'; readonly config: WorkerConfig }
  | { readonly type: 'warm-up' }
  | { readonly type: 'rewrite'; readonly jobId: string; readonly style: StyleId; readonly text: string }
  | { readonly type: 'cancel'; readonly jobId: string }
  | { readonly type: 'download' }
  | { readonly type: 'cancel-download' }
  | { readonly type: 'remove-model' };

/** What the engine reports; relayed unchanged from the worker up to the service worker. */
export type EngineEvent =
  | { readonly type: 'status'; readonly status: EngineStatus }
  | { readonly type: 'job-phase'; readonly jobId: string; readonly phase: JobPhase; readonly progress?: number }
  | { readonly type: 'job-done'; readonly jobId: string; readonly text: string }
  | { readonly type: 'job-failed'; readonly jobId: string; readonly error: ErrorPayload }
  | { readonly type: 'installed'; readonly installed: InstalledModelRecord | null }
  | { readonly type: 'download'; readonly download: DownloadState };

// ---------------------------------------------------------------------------
// extension pages (popup, welcome) → service worker
// ---------------------------------------------------------------------------

const UI_REQUESTS = ['ui/warm-up', 'ui/download-model', 'ui/cancel-download', 'ui/remove-model'] as const;

export type UiRequest = { readonly kind: (typeof UI_REQUESTS)[number] };

export type UiReply = { readonly ok: true } | { readonly ok: false; readonly error: ErrorPayload };

export function isUiRequest(value: unknown): value is UiRequest {
  return typeof value === 'object' && value !== null && (UI_REQUESTS as readonly unknown[]).includes((value as { kind?: unknown }).kind);
}
