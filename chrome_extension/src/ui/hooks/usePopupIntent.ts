import { useEffect, useState } from 'react';
import { takePopupIntent, type PopupIntent } from '../../shared/popupIntent';

let taken: Promise<PopupIntent | null> | null = null;

/** Taken once per page load: a second mount (React's strict mode) must not find it gone and start on the home page. */
function takeOnce(): Promise<PopupIntent | null> {
  taken ??= takePopupIntent().catch(() => null);
  return taken;
}

/** Where the popup was asked to open; undefined until that is known (a few milliseconds), null for its home page. */
export function usePopupIntent(): PopupIntent | null | undefined {
  const [intent, setIntent] = useState<PopupIntent | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    void takeOnce().then((value) => live && setIntent(value));
    return () => {
      live = false;
    };
  }, []);
  return intent;
}
