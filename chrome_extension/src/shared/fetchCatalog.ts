import { catalogOf, parseModelManifest } from '../engine/manifest';
import type { ModelCatalog } from './modelInstall';
import { MODEL_SOURCE_URL } from './modelSource';

/** What the download server offers right now (model.json, read fresh). */
export async function fetchCatalog(fetchImpl: typeof fetch = fetch): Promise<ModelCatalog> {
  const response = await fetchImpl(new URL('model.json', MODEL_SOURCE_URL), { cache: 'no-store' });
  if (!response.ok) throw new Error(`model.json: HTTP ${response.status}`);
  return catalogOf(parseModelManifest(await response.json()));
}
