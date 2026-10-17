/**
 * The download server's catalog, as the service worker last read it: kept
 * for a while, so asking costs nothing on most rewrites, and given up on
 * quickly, so a slow or missing server never holds a rewrite up for long.
 */
import type { ModelCatalog } from '../shared/modelInstall';

const KEEP_MS = 10 * 60_000;
/** After a failure, try the server again sooner. */
const RETRY_MS = 60_000;
const TIMEOUT_MS = 2_500;

export interface RemoteCatalogDeps {
  readonly fetchCatalog: () => Promise<ModelCatalog>;
  readonly now?: () => number;
}

export function remoteCatalogCache(deps: RemoteCatalogDeps): () => Promise<ModelCatalog | null> {
  const now = deps.now ?? Date.now;
  let kept: { catalog: ModelCatalog | null; until: number } | null = null;
  let pending: Promise<ModelCatalog | null> | null = null;
  return () => {
    if (kept && now() < kept.until) return Promise.resolve(kept.catalog);
    pending ??= Promise.race([
      deps.fetchCatalog(),
      new Promise<never>((_, reject) => setTimeout(() => reject(new Error('timeout')), TIMEOUT_MS)),
    ]).then(
      (catalog) => {
        kept = { catalog, until: now() + KEEP_MS };
        return catalog;
      },
      () => {
        kept = { catalog: null, until: now() + RETRY_MS };
        return null;
      },
    ).finally(() => {
      pending = null;
    });
    return pending;
  };
}
