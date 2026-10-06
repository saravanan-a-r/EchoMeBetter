import { useEffect, useState } from 'react';
import { INITIAL_STATUS, type EngineStatus } from '../../shared/status';
import { readStatus, subscribeStatus } from '../../shared/statusStore';

/** Live engine status, read from session storage and kept current. */
export function useEngineStatus(): EngineStatus {
  const [status, setStatus] = useState<EngineStatus>(INITIAL_STATUS);
  useEffect(() => {
    let alive = true;
    void readStatus().then((current) => {
      if (alive) setStatus(current);
    });
    const unsubscribe = subscribeStatus(setStatus);
    return () => {
      alive = false;
      unsubscribe();
    };
  }, []);
  return status;
}

export function requestWarmUp(): void {
  void chrome.runtime.sendMessage({ kind: 'ui/warm-up' });
}
