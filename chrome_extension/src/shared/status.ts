/**
 * The model's lifecycle as the UI sees it.
 *
 * The service worker is the only writer (it hears every transition from the
 * offscreen document); the popup and welcome page read it from
 * `chrome.storage.session`, so they survive the service worker being
 * suspended between events.
 */
export type EngineStatus =
  | { readonly state: 'unloaded' }
  | { readonly state: 'loading'; readonly progress: number }
  | { readonly state: 'ready'; readonly model: ModelSummary }
  | { readonly state: 'error'; readonly message: string };

export interface ModelSummary {
  readonly id: string;
  readonly displayName: string;
  readonly placeholder: boolean;
  readonly precision: string;
  readonly sizeBytes: number;
}

export const STATUS_STORAGE_KEY = 'engineStatus';

export const INITIAL_STATUS: EngineStatus = { state: 'unloaded' };

export function isEngineStatus(value: unknown): value is EngineStatus {
  if (typeof value !== 'object' || value === null) return false;
  const status = value as { state?: unknown };
  return status.state === 'unloaded' || status.state === 'loading' || status.state === 'ready' || status.state === 'error';
}
