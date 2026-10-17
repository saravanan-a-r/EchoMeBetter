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
import type { ComputeSettings, GpuProblem } from './compute';
import { isErrorPayload, type ErrorPayload } from './errors';
import { isSettingsFocus, type SettingsFocus } from './popupIntent';
import type { KeepLoadedMinutes } from './settings';
import type { KeyPlatform } from './shortcuts';
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
  /** The style the user asked for, when that is what the notice is about (a missing model offers to carry on with it). */
  readonly style?: StyleId;
}

/** Open the style menu at `point` (the frame's viewport coordinates), for the selection a press and hold was made on. */
export interface StyleMenuMessage {
  readonly kind: 'echo/style-menu';
  readonly point: Point;
}

export interface Point {
  readonly x: number;
  readonly y: number;
}

export type ForegroundMessage = StartJobMessage | NoticeMessage | StyleMenuMessage;

export type StartJobReply = { readonly ok: true } | { readonly ok: false; readonly error: ErrorPayload };

export function isForegroundMessage(value: unknown): value is ForegroundMessage {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as { kind?: unknown; jobId?: unknown; style?: unknown; error?: unknown };
  if (message.kind === 'echo/start-job') return typeof message.jobId === 'string' && isStyleId(message.style);
  if (message.kind === 'echo/notice') return typeof message.error === 'object' && message.error !== null;
  if (message.kind === 'echo/style-menu') return isPoint((value as { point?: unknown }).point);
  return false;
}

function isPoint(value: unknown): value is Point {
  if (typeof value !== 'object' || value === null) return false;
  const { x, y } = value as { x?: unknown; y?: unknown };
  return typeof x === 'number' && Number.isFinite(x) && typeof y === 'number' && Number.isFinite(y);
}

// ---------------------------------------------------------------------------
// page listener (shortcuts, press and hold; any page) and style menu → service worker
// ---------------------------------------------------------------------------

/** A style's shortcut was pressed with text selected in an editable field of the sender's frame. */
export interface ShortcutRequest {
  readonly kind: 'shortcut/rewrite';
  readonly style: StyleId;
}

export function isShortcutRequest(value: unknown): value is ShortcutRequest {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as { kind?: unknown; style?: unknown };
  return message.kind === 'shortcut/rewrite' && isStyleId(message.style);
}

/**
 * The style menu, in the sender's frame:
 *   menu/open      text was pressed and held at `point`: show the menu there
 *   menu/rewrite   a style was picked from it
 */
export type StyleMenuRequest = { readonly kind: 'menu/open'; readonly point: Point } | { readonly kind: 'menu/rewrite'; readonly style: StyleId };

export function isStyleMenuRequest(value: unknown): value is StyleMenuRequest {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as { kind?: unknown; style?: unknown; point?: unknown };
  if (message.kind === 'menu/open') return isPoint(message.point);
  return message.kind === 'menu/rewrite' && isStyleId(message.style);
}

// ---------------------------------------------------------------------------
// foreground (a toast's button) → service worker
// ---------------------------------------------------------------------------

/** Open the toolbar popup on its settings page, at `focus`. */
export interface OpenSettingsRequest {
  readonly kind: 'page/open-settings';
  readonly focus: SettingsFocus;
}

export function isOpenSettingsRequest(value: unknown): value is OpenSettingsRequest {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as { kind?: unknown; focus?: unknown };
  return message.kind === 'page/open-settings' && isSettingsFocus(message.focus);
}

/**
 * A rewrite was just put in place. The service worker counts it and says
 * whether to offer the user the site access that hold and shortcuts need.
 */
export interface ReplacedNotice {
  readonly kind: 'access/replaced';
}

export function isReplacedNotice(value: unknown): value is ReplacedNotice {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'access/replaced';
}

/** What the offer is for: the features that wait for the access, and the keys to show. */
export interface AccessOffer {
  readonly hold: boolean;
  readonly shortcuts: boolean;
  readonly platform: KeyPlatform;
}

export type AccessOfferReply = { readonly offer: false } | ({ readonly offer: true } & AccessOffer);

/** The user pressed Allow on the offer: show Chrome's permission prompt. */
export interface AccessRequest {
  readonly kind: 'access/request';
}

export function isAccessRequest(value: unknown): value is AccessRequest {
  return typeof value === 'object' && value !== null && (value as { kind?: unknown }).kind === 'access/request';
}

/**
 *   granted   the user allowed websites
 *   denied    the user said no in Chrome's prompt
 *   opened    Chrome would not show the prompt from here, so the popup opened on the button that does
 */
export type AccessResult = 'granted' | 'denied' | 'opened';

// ---------------------------------------------------------------------------
// foreground ⇄ service worker, over a Port named SETUP_PORT_NAME (the model download, in place)
// ---------------------------------------------------------------------------

export const SETUP_PORT_NAME = 'echo/setup';

/** What a rewrite is missing: the writing model (with its first style), or one style on top of it. */
export type SetupNeed = { readonly kind: 'model'; readonly style?: StyleId } | { readonly kind: 'style'; readonly style: StyleId };

export type SetupRequest =
  /** The card opened: report what the download would be, or how it is going. */
  | { readonly kind: 'setup/open'; readonly need: SetupNeed }
  /** Start the download (also the retry of one that failed). */
  | { readonly kind: 'setup/download' };

/** Where the download stands, for the card to show. */
export type SetupEvent =
  /** Nothing is downloading: this is what a download would fetch. */
  | { readonly kind: 'setup/offer'; readonly totalBytes: number; /** The styles the download brings, the one asked for first. */ readonly styles?: readonly StyleId[] }
  | { readonly kind: 'setup/progress'; readonly verifying: boolean; readonly receivedBytes: number; readonly totalBytes: number; readonly bytesPerSecond: number }
  | { readonly kind: 'setup/done' }
  | { readonly kind: 'setup/failed'; readonly error: ErrorPayload };

function isSetupNeed(value: unknown): value is SetupNeed {
  if (typeof value !== 'object' || value === null) return false;
  const need = value as { kind?: unknown; style?: unknown };
  return (need.kind === 'model' && (need.style === undefined || isStyleId(need.style))) || (need.kind === 'style' && isStyleId(need.style));
}

export function isSetupRequest(value: unknown): value is SetupRequest {
  if (typeof value !== 'object' || value === null) return false;
  const message = value as { kind?: unknown; need?: unknown };
  return message.kind === 'setup/download' || (message.kind === 'setup/open' && isSetupNeed(message.need));
}

export function isSetupEvent(value: unknown): value is SetupEvent {
  if (typeof value !== 'object' || value === null) return false;
  const event = value as { kind?: unknown; styles?: unknown; totalBytes?: unknown; receivedBytes?: unknown; error?: unknown; verifying?: unknown; bytesPerSecond?: unknown };
  switch (event.kind) {
    case 'setup/offer':
      return typeof event.totalBytes === 'number' && (event.styles === undefined || (Array.isArray(event.styles) && event.styles.every(isStyleId)));
    case 'setup/progress':
      return typeof event.verifying === 'boolean' && typeof event.receivedBytes === 'number' && typeof event.totalBytes === 'number' && typeof event.bytesPerSecond === 'number';
    case 'setup/done':
      return true;
    case 'setup/failed':
      return isErrorPayload(event.error);
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// foreground ⇄ service worker, over a Port named JOB_PORT_NAME (one per job)
// ---------------------------------------------------------------------------

export type JobRequest =
  | { readonly kind: 'job/request'; readonly jobId: string; readonly style: StyleId; readonly text: string }
  | { readonly kind: 'job/cancel'; readonly jobId: string };

export type JobPhase = 'loading-model' | 'rewriting';

/**
 * The model had been put to rest by the idle timer, and this job woke it up.
 * `idleMinutes` is the user's current keep-awake time, which a longer one
 * would have avoided.
 */
export interface WokeFromRest {
  readonly idleMinutes: Exclude<KeepLoadedMinutes, 0>;
}

export type JobEvent =
  | { readonly kind: 'job/phase'; readonly jobId: string; readonly phase: JobPhase; readonly progress?: number }
  /** The rewrite so far, while the model writes it (only with the live preview on). Not the result: job/done carries that. */
  | { readonly kind: 'job/partial'; readonly jobId: string; readonly text: string }
  | { readonly kind: 'job/done'; readonly jobId: string; readonly text: string; readonly wokeFromRest?: WokeFromRest }
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
  return kind === 'job/phase' || kind === 'job/partial' || kind === 'job/done' || kind === 'job/failed';
}

// ---------------------------------------------------------------------------
// service worker ⇄ offscreen document (chrome.runtime.sendMessage)
// ---------------------------------------------------------------------------

export type OffscreenCommand =
  /** Sent once, right after the document is created: starts the worker. The reply proves the round trip. */
  | { readonly target: 'offscreen'; readonly kind: 'engine/start'; readonly compute: ComputeSettings }
  | { readonly target: 'offscreen'; readonly kind: 'engine/warm-up' }
  | { readonly target: 'offscreen'; readonly kind: 'engine/rewrite'; readonly jobId: string; readonly style: StyleId; readonly text: string; readonly stream: boolean }
  | { readonly target: 'offscreen'; readonly kind: 'engine/cancel'; readonly jobId: string }
  | { readonly target: 'offscreen'; readonly kind: 'model/download'; readonly adapters: readonly StyleId[] }
  | { readonly target: 'offscreen'; readonly kind: 'model/cancel-download' }
  | { readonly target: 'offscreen'; readonly kind: 'model/remove-adapter'; readonly adapter: StyleId }
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
  /** Absolute URL of the folder holding onnxruntime-web's .wasm binaries. */
  readonly wasmBaseUrl: string;
  /** Fixed for the worker's lifetime: onnxruntime reads it once. The service worker restarts the worker to change it. */
  readonly compute: ComputeSettings;
}

export type WorkerRequest =
  | { readonly type: 'configure'; readonly config: WorkerConfig }
  | { readonly type: 'warm-up' }
  /** `stream`: report the rewrite as it is written (job-partial), for the live preview. */
  | { readonly type: 'rewrite'; readonly jobId: string; readonly style: StyleId; readonly text: string; readonly stream: boolean }
  | { readonly type: 'cancel'; readonly jobId: string }
  | { readonly type: 'download'; readonly adapters: readonly StyleId[] }
  | { readonly type: 'cancel-download' }
  | { readonly type: 'remove-adapter'; readonly adapter: StyleId }
  | { readonly type: 'remove-model' };

/** What the engine reports; relayed unchanged from the worker up to the service worker. */
export type EngineEvent =
  | { readonly type: 'status'; readonly status: EngineStatus }
  | { readonly type: 'job-phase'; readonly jobId: string; readonly phase: JobPhase; readonly progress?: number }
  | { readonly type: 'job-partial'; readonly jobId: string; readonly text: string }
  | { readonly type: 'job-done'; readonly jobId: string; readonly text: string }
  | { readonly type: 'job-failed'; readonly jobId: string; readonly error: ErrorPayload }
  | { readonly type: 'installed'; readonly installed: InstalledModelRecord | null }
  | { readonly type: 'download'; readonly download: DownloadState }
  /** The GPU could not run the model at all; the engine moved to the processor. */
  | { readonly type: 'gpu-problem'; readonly problem: GpuProblem };

// ---------------------------------------------------------------------------
// extension pages (popup, welcome) → service worker
// ---------------------------------------------------------------------------

export type UiRequest =
  | { readonly kind: 'ui/warm-up' }
  /** Download these adapters, and the base first if it is not installed. */
  | { readonly kind: 'ui/download-model'; readonly adapters: readonly StyleId[] }
  | { readonly kind: 'ui/cancel-download' }
  | { readonly kind: 'ui/remove-adapter'; readonly adapter: StyleId }
  /** The base and every adapter. */
  | { readonly kind: 'ui/remove-model' };

export type UiReply = { readonly ok: true } | { readonly ok: false; readonly error: ErrorPayload };

export function isUiRequest(value: unknown): value is UiRequest {
  if (typeof value !== 'object' || value === null) return false;
  const request = value as { kind?: unknown; adapters?: unknown; adapter?: unknown };
  switch (request.kind) {
    case 'ui/warm-up':
    case 'ui/cancel-download':
    case 'ui/remove-model':
      return true;
    case 'ui/download-model':
      return Array.isArray(request.adapters) && request.adapters.every(isStyleId);
    case 'ui/remove-adapter':
      return isStyleId(request.adapter);
    default:
      return false;
  }
}
