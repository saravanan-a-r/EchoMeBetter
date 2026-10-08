import { useEffect, useState } from 'react';
import { detectPlatform, SITE_ACCESS, type KeyPlatform } from '../../shared/shortcuts';

/**
 * Where a feature that works on websites (the keyboard shortcuts, the press
 * and hold menu) stands, as the UI shows it:
 *   loading        not read yet; show nothing
 *   off            the user turned it off
 *   needs-access   turned on, but websites are not allowed yet
 *   on             working on websites
 */
export type WebsiteAvailability = 'loading' | 'off' | 'needs-access' | 'on';

export function websiteAvailability(enabled: boolean | null, siteAccess: boolean | null): WebsiteAvailability {
  if (enabled === null || siteAccess === null) return 'loading';
  if (!enabled) return 'off';
  return siteAccess ? 'on' : 'needs-access';
}

export const PLATFORM: KeyPlatform = detectPlatform(navigator as Navigator & { userAgentData?: { platform?: string } });

const origins = () => ({ origins: [...SITE_ACCESS.origins] });

/** Whether the user has allowed EchoMeBetter on websites; null until known. */
export function useSiteAccess(): boolean | null {
  const [granted, setGranted] = useState<boolean | null>(null);
  useEffect(() => {
    let live = true;
    const check = () => {
      chrome.permissions
        .contains(origins())
        .then((value) => live && setGranted(value))
        .catch(() => live && setGranted(false));
    };
    check();
    chrome.permissions.onAdded.addListener(check);
    chrome.permissions.onRemoved.addListener(check);
    return () => {
      live = false;
      chrome.permissions.onAdded.removeListener(check);
      chrome.permissions.onRemoved.removeListener(check);
    };
  }, []);
  return granted;
}

/** Shows Chrome's permission prompt; must run straight from a click. */
export function requestSiteAccess(): void {
  chrome.permissions.request(origins()).catch(() => undefined);
}

export function releaseSiteAccess(): void {
  chrome.permissions.remove(origins()).catch(() => undefined);
}

/** Whether either feature waits for the user to allow websites. */
export function needsSiteAccess(...features: readonly WebsiteAvailability[]): boolean {
  return features.includes('needs-access');
}
