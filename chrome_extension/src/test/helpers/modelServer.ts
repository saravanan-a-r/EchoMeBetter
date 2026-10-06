/**
 * A `fetch` that serves a model folder from memory the way a static HTTP
 * server does: 200 for a whole file, 206 for `Range: bytes=<start>-`, 404
 * for anything else. Knobs simulate the failures a real download meets.
 */
import { readFixture } from './fixtures';

export const FIXTURE_URL = 'https://models.example.test/tiny/';

export interface ServerOptions {
  /** Answer range requests with the whole file, as some servers do. */
  readonly ignoreRange?: boolean;
  /** Bytes per body chunk. */
  readonly chunkSize?: number;
  /** Drop the connection after this many body bytes of the named file. */
  readonly dropAfter?: { readonly path: string; readonly bytes: number };
}

export interface ServedRequest {
  readonly url: string;
  readonly range: string | null;
  readonly cache: RequestCache | undefined;
}

/** A tiny test model (int8 by default), as the files a server would hold. */
export function tinyModelFiles(fixture: 'tiny-t5-int8' | 'tiny-t5' = 'tiny-t5-int8'): Map<string, Uint8Array> {
  return new Map(
    ['model.json', 'encoder.onnx', 'decoder.onnx', 'tokenizer.json'].map((name) => [name, new Uint8Array(readFixture(`${fixture}/${name}`))]),
  );
}

function abortError(): DOMException {
  return new DOMException('The operation was aborted.', 'AbortError');
}

export function modelServer(files: Map<string, Uint8Array>, options: ServerOptions = {}) {
  const requests: ServedRequest[] = [];
  const chunkSize = options.chunkSize ?? 4096;

  const fetch = async (url: string, init: RequestInit = {}): Promise<Response> => {
    const signal = init.signal ?? undefined;
    if (signal?.aborted) throw abortError();
    const range = new Headers(init.headers).get('range');
    requests.push({ url, range, cache: init.cache });
    if (!url.startsWith(FIXTURE_URL)) throw new TypeError('Failed to fetch');
    const path = url.slice(FIXTURE_URL.length);
    const data = files.get(path);
    if (!data) return new Response('not found', { status: 404 });

    const match = range && !options.ignoreRange ? /^bytes=(\d+)-$/.exec(range) : null;
    const start = match ? Number(match[1]) : 0;
    const body = data.subarray(start);
    const dropAt = options.dropAfter?.path === path ? options.dropAfter.bytes : Number.POSITIVE_INFINITY;
    let sent = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (signal?.aborted) return controller.error(abortError());
        if (sent >= dropAt) return controller.error(new TypeError('network connection was lost'));
        if (sent >= body.byteLength) return controller.close();
        const end = Math.min(body.byteLength, sent + chunkSize, dropAt);
        controller.enqueue(body.slice(sent, end));
        sent = end;
      },
    });
    const headers: Record<string, string> = { 'content-length': String(body.byteLength) };
    if (match) headers['content-range'] = `bytes ${start}-${data.byteLength - 1}/${data.byteLength}`;
    return new Response(stream, { status: match ? 206 : 200, headers });
  };

  return { fetch, requests };
}
