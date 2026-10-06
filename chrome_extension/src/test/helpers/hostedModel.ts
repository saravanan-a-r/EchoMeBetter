/**
 * Jest global setup: is the model in model.source.json being served?
 *
 * Developers host the model themselves (any static server with CORS and
 * range support) at the URL in model.source.json. When it answers, its URL
 * is handed to the tests that need the real model through
 * ECHOMEBETTER_HOSTED_MODEL; when it does not, those tests are skipped and
 * every other test runs as usual.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

export const HOSTED_MODEL_ENV = 'ECHOMEBETTER_HOSTED_MODEL';

export default async function probeHostedModel(): Promise<void> {
  const { url } = JSON.parse(readFileSync(join(__dirname, '..', '..', '..', 'model.source.json'), 'utf8')) as { url: string };
  try {
    const response = await fetch(new URL('parity.json', url), { signal: AbortSignal.timeout(2000) });
    await response.body?.cancel();
    if (response.ok) process.env[HOSTED_MODEL_ENV] = url;
  } catch {
    // Not served: the real-model tests are skipped.
  }
}
