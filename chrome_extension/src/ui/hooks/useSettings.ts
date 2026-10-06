import { useCallback, useEffect, useState } from 'react';
import { DEFAULT_SETTINGS, loadSettings, parseSettings, saveSettings, SETTINGS_STORAGE_KEY, type Settings } from '../../shared/settings';

/** The settings, and whether they have been read from storage yet. Follows changes made from other pages. */
export function useSettings(): [Settings, (next: Settings) => void, boolean] {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    void loadSettings().then((stored) => {
      setSettings(stored);
      setLoaded(true);
    });
    const onChanged = (changes: Record<string, chrome.storage.StorageChange>, area: string) => {
      const change = changes[SETTINGS_STORAGE_KEY];
      if (area === 'local' && change) setSettings(parseSettings(change.newValue));
    };
    chrome.storage.onChanged.addListener(onChanged);
    return () => chrome.storage.onChanged.removeListener(onChanged);
  }, []);
  const update = useCallback((next: Settings) => {
    setSettings(next);
    void saveSettings(next);
  }, []);
  return [settings, update, loaded];
}
