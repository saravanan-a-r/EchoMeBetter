import { useEffect, useState } from 'react';
import { findGpu, GPU_PROBLEM_STORAGE_KEY, parseGpuProblem, readGpuProblem, writeGpuProblem, type GpuInfo, type GpuLike, type GpuProblem } from '../../shared/compute';

/** Whether this browser offers a GPU the model can run on; asked once per page. */
export type GpuAvailability = { readonly state: 'checking' } | { readonly state: 'none' } | { readonly state: 'available'; readonly gpu: GpuInfo };

export function useGpu(): GpuAvailability {
  const [availability, setAvailability] = useState<GpuAvailability>({ state: 'checking' });
  useEffect(() => {
    let alive = true;
    void findGpu((navigator as { gpu?: GpuLike }).gpu).then((gpu) => {
      if (alive) setAvailability(gpu ? { state: 'available', gpu } : { state: 'none' });
    });
    return () => {
      alive = false;
    };
  }, []);
  return availability;
}

/** A GPU recorded as unable to run the model, kept current. */
export function useGpuProblem(): GpuProblem | null {
  const [problem, setProblem] = useState<GpuProblem | null>(null);
  useEffect(() => {
    let alive = true;
    void readGpuProblem().then((stored) => {
      if (alive) setProblem(stored);
    });
    const onChanged = (changes: Record<string, chrome.storage.StorageChange>, area: string) => {
      const change = changes[GPU_PROBLEM_STORAGE_KEY];
      if (area === 'local' && change) setProblem(parseGpuProblem(change.newValue));
    };
    chrome.storage.onChanged.addListener(onChanged);
    return () => {
      alive = false;
      chrome.storage.onChanged.removeListener(onChanged);
    };
  }, []);
  return problem;
}

/** Forget the recorded problem; the service worker then starts the model on the GPU again. */
export function retryGpu(): void {
  void writeGpuProblem(null);
}

/** The computer the model's threads are counted against. Extension pages are cross-origin isolated, like the worker. */
export const MACHINE = {
  cores: typeof navigator === 'undefined' ? 1 : navigator.hardwareConcurrency,
  crossOriginIsolated: typeof globalThis.crossOriginIsolated === 'boolean' ? globalThis.crossOriginIsolated : false,
} as const;
