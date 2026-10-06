/**
 * Download model files from the folder at `sourceUrl` into the ModelStore.
 *
 *   1. model.json: validated in full; it names every file with its size and sha256
 *   2. enough free space for what is still missing, or a clear error
 *   3. each file, resuming whatever an earlier attempt left on disk (HTTP Range)
 *   4. each file checked against its sha256; a mismatch deletes it
 *
 * Which files to fetch (the base, adapters) and recording them as installed
 * is the caller's part (modelLibrary.ts).
 *
 * Responses bypass the HTTP cache (`no-store`), or Chrome would keep a second
 * copy of every file there.
 */
import { EchoError } from '../shared/errors';
import { parseModelManifest, type ModelFile, type ModelManifest } from '../engine/manifest';
import type { ModelStore } from './modelStore';

export interface DownloadProgress {
  readonly phase: 'fetching' | 'verifying';
  readonly receivedBytes: number;
  readonly totalBytes: number;
  readonly bytesPerSecond: number;
}

export interface DownloadTransport {
  readonly fetch: (url: string, init: RequestInit) => Promise<Response>;
  readonly estimate: () => Promise<{ quota?: number; usage?: number }>;
  /** Lowercase hex SHA-256 of `bytes`. */
  readonly digest: (bytes: Uint8Array) => Promise<string>;
  readonly now: () => number;
}

export interface DownloadOptions {
  readonly signal: AbortSignal;
  readonly onProgress: (progress: DownloadProgress) => void;
}

/** Headroom left free beyond the model itself. */
const SPACE_MARGIN = 64 * 1024 * 1024;
/** Written data is flushed to disk this often, so little is lost to a crash. */
const FLUSH_EVERY = 64 * 1024 * 1024;
const REPORT_EVERY_MS = 250;
const SPEED_WINDOW_MS = 3000;

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes as Uint8Array<ArrayBuffer>));
  return Array.from(hash, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/** Throttled progress, with the speed measured over the last few seconds. */
class ProgressMeter {
  private received: number;
  private lastReport = Number.NEGATIVE_INFINITY;
  private readonly samples: { at: number; received: number }[] = [];

  constructor(
    private readonly total: number,
    alreadyOnDisk: number,
    private readonly now: () => number,
    private readonly report: (progress: DownloadProgress) => void,
  ) {
    this.received = alreadyOnDisk;
    this.samples.push({ at: now(), received: alreadyOnDisk });
  }

  add(count: number): void {
    this.received += count;
    const at = this.now();
    if (at - this.lastReport < REPORT_EVERY_MS) return;
    this.samples.push({ at, received: this.received });
    while (this.samples.length > 2 && at - this.samples[0]!.at > SPEED_WINDOW_MS) this.samples.shift();
    this.emit('fetching', at);
  }

  verifying(): void {
    this.emit('verifying', this.now());
  }

  private emit(phase: DownloadProgress['phase'], at: number): void {
    this.lastReport = at;
    const first = this.samples[0]!;
    const elapsed = at - first.at;
    const bytesPerSecond = phase === 'fetching' && elapsed > 0 ? Math.max(0, ((this.received - first.received) / elapsed) * 1000) : 0;
    this.report({ phase, receivedBytes: this.received, totalBytes: this.total, bytesPerSecond });
  }
}

function failed(reason: string, details: Record<string, string | number> = {}): EchoError {
  return new EchoError('DOWNLOAD_FAILED', { reason, ...details });
}

async function request(transport: DownloadTransport, url: URL, signal: AbortSignal, headers?: Record<string, string>): Promise<Response> {
  const response = await transport.fetch(url.toString(), { headers, signal, cache: 'no-store' });
  if (!response.ok) throw failed(`HTTP ${response.status}`, { url: url.toString() });
  return response;
}

async function ensureSpace(transport: DownloadTransport, neededBytes: number): Promise<void> {
  if (neededBytes <= 0) return;
  const { quota, usage } = await transport.estimate();
  if (quota === undefined || usage === undefined) return;
  if (quota - usage < neededBytes + SPACE_MARGIN) throw new EchoError('STORAGE_FULL', { neededBytes: neededBytes + SPACE_MARGIN - (quota - usage) });
}

/** The first byte a 206 response starts at, from `Content-Range: bytes <start>-<end>/<size>`. */
function rangeStart(response: Response): number | null {
  const match = /^bytes (\d+)-/.exec(response.headers.get('content-range') ?? '');
  return match ? Number(match[1]) : null;
}

async function fetchFile(
  store: ModelStore,
  sourceUrl: string,
  file: ModelFile,
  transport: DownloadTransport,
  signal: AbortSignal,
  onBytes: (count: number) => void,
): Promise<void> {
  const handle = await store.openForWrite(file);
  try {
    let at = handle.getSize();
    if (at === file.bytes) return;
    if (at > file.bytes) {
      handle.truncate(0);
      at = 0;
    }
    const url = new URL(file.path, sourceUrl);
    const response = await request(transport, url, signal, at > 0 ? { Range: `bytes=${at}-` } : undefined);
    if (at > 0 && response.status !== 206) {
      // The server sent the whole file instead of the rest of it: start this file over.
      onBytes(-at);
      handle.truncate(0);
      at = 0;
    } else if (at > 0 && rangeStart(response) !== at) {
      throw failed('the server answered a resume request with the wrong range', { file: file.path });
    }
    if (!response.body) throw failed('empty response', { file: file.path });

    const reader = response.body.getReader();
    let unflushed = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (at + value.byteLength > file.bytes) {
        await reader.cancel();
        throw failed('file is larger than model.json declares', { file: file.path });
      }
      handle.write(value, { at });
      at += value.byteLength;
      unflushed += value.byteLength;
      onBytes(value.byteLength);
      if (unflushed >= FLUSH_EVERY) {
        handle.flush();
        unflushed = 0;
      }
    }
    if (at !== file.bytes) throw failed('file is truncated', { file: file.path, expected: file.bytes, received: at });
  } finally {
    handle.flush();
    handle.close();
  }
}

/** model.json from `sourceUrl`, validated. */
export async function fetchManifest(sourceUrl: string, transport: DownloadTransport, signal: AbortSignal): Promise<ModelManifest> {
  const raw: unknown = await (await request(transport, new URL('model.json', sourceUrl), signal)).json();
  try {
    return parseModelManifest(raw);
  } catch (error) {
    throw failed(error instanceof Error ? error.message : String(error));
  }
}

async function download(store: ModelStore, sourceUrl: string, files: readonly ModelFile[], transport: DownloadTransport, options: DownloadOptions): Promise<void> {
  const { signal } = options;
  const total = files.reduce((sum, file) => sum + file.bytes, 0);
  // A file longer than declared is restarted by fetchFile, so it counts as nothing on disk.
  const onDisk = await Promise.all(
    files.map(async (file) => {
      const size = await store.size(file);
      return size > file.bytes ? 0 : size;
    }),
  );
  const alreadyOnDisk = onDisk.reduce((sum, size) => sum + size, 0);
  await ensureSpace(transport, total - alreadyOnDisk);

  const meter = new ProgressMeter(total, alreadyOnDisk, transport.now, options.onProgress);
  for (const file of files) await fetchFile(store, sourceUrl, file, transport, signal, (count) => meter.add(count));

  meter.verifying();
  for (const file of files) {
    if (signal.aborted) throw new EchoError('CANCELLED');
    if ((await transport.digest(await store.read(file))) !== file.sha256) {
      await store.remove(file);
      throw failed('checksum mismatch', { file: file.path });
    }
  }
}

/** Fetch and verify `files` (listed in the model.json at `sourceUrl`). */
export async function downloadFiles(
  store: ModelStore,
  sourceUrl: string,
  files: readonly ModelFile[],
  transport: DownloadTransport,
  options: DownloadOptions,
): Promise<void> {
  try {
    await download(store, sourceUrl, files, transport, options);
  } catch (error) {
    throw downloadError(error, options.signal);
  }
}

/** Any failure of a download, as the error the user is shown. */
export function downloadError(error: unknown, signal: AbortSignal): EchoError {
  if (signal.aborted) return new EchoError('CANCELLED');
  if (error instanceof EchoError) return error;
  if (error instanceof DOMException && error.name === 'QuotaExceededError') return new EchoError('STORAGE_FULL');
  // fetch rejects with a TypeError when the network or CORS fails.
  return failed(error instanceof Error ? error.message : String(error));
}
