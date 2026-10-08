import { useSyncExternalStore } from 'react';
import { CursorLoader } from './CursorLoader';
import type { OverlayStore } from './overlayStore';
import { StyleMenu } from './StyleMenu';
import { Toast } from './Toast';

export function OverlayApp({ store }: { store: OverlayStore }) {
  const { working, toast, menu } = useSyncExternalStore(store.subscribe, store.getSnapshot);
  return (
    <>
      {menu ? <StyleMenu menu={menu} /> : null}
      {working ? <CursorLoader working={working} /> : null}
      {toast ? <Toast key={toast.id} toast={toast} onDismiss={() => store.dismissToast(toast.id)} /> : null}
    </>
  );
}
