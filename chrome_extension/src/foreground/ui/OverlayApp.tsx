import { useSyncExternalStore } from 'react';
import { CursorLoader } from './CursorLoader';
import { OverlayCard } from './OverlayCard';
import type { OverlayStore } from './overlayStore';
import { StyleMenu } from './StyleMenu';

export function OverlayApp({ store }: { store: OverlayStore }) {
  const { working, toast, menu } = useSyncExternalStore(store.subscribe, store.getSnapshot);
  // Until the first words arrive the pointer loader shows the rewrite is under way; from then on the card does.
  const streaming = working?.preview ? { ...working, preview: working.preview } : null;
  return (
    <>
      {menu ? <StyleMenu menu={menu} /> : null}
      {working && !streaming ? <CursorLoader working={working} /> : null}
      {/* A toast raised meanwhile (one rewrite at a time) has the card until it closes; the preview then carries on. */}
      {toast || streaming ? <OverlayCard toast={toast} streaming={streaming} onDismiss={() => toast && store.dismissToast(toast.id)} /> : null}
    </>
  );
}
