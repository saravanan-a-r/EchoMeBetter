/**
 * Where the extension downloads its model from: the `url` in
 * model.source.json, at the project root.
 *
 * The URL names a folder holding model.json and the files it lists. The
 * extension downloads them once, when the user asks, into its own private
 * storage (see worker/modelStore.ts); nothing model-sized ships in the package.
 *
 * Every file is checked against the sha256 in model.json, so the folder must
 * be served over HTTPS -- plain HTTP is accepted only from this machine, for
 * development.
 */
import source from '../../model.source.json';

const LOOPBACK_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

export function parseModelSourceUrl(raw: unknown): string {
  if (typeof raw !== 'string' || raw.length === 0) throw new Error('model.source.json: "url" must be a non-empty string');
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`model.source.json: "${raw}" is not a valid URL`);
  }
  const secure = url.protocol === 'https:' || (url.protocol === 'http:' && LOOPBACK_HOSTS.has(url.hostname));
  if (!secure) throw new Error('model.source.json: the model must be served over https (plain http only from localhost)');
  if (url.search || url.hash) throw new Error('model.source.json: the URL must not have a query or fragment');
  if (!url.pathname.endsWith('/')) throw new Error('model.source.json: the URL names a folder, so it must end with "/"');
  return url.toString();
}

export const MODEL_SOURCE_URL = parseModelSourceUrl(source.url);
