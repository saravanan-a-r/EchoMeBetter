/** Numbers as people read them in the extension's UI. */

export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${Math.round(bytes / 1e6)} MB`;
  return `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

export function formatTimeLeft(seconds: number): string {
  if (seconds < 60) return `about ${Math.max(1, Math.round(seconds))} s left`;
  if (seconds < 3600) return `about ${Math.round(seconds / 60)} min left`;
  const hours = Math.floor(seconds / 3600);
  return `about ${hours} h ${Math.round((seconds - hours * 3600) / 60)} min left`;
}

/** How long a rewrite took: milliseconds under a second, seconds under a minute, then minutes. */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.max(1, Math.round(ms))} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms - minutes * 60_000) / 1000);
  return seconds === 0 ? `${minutes} min` : `${minutes} min ${seconds} s`;
}
