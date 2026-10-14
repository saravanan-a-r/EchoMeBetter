import { useSyncExternalStore } from 'react';
import { CursorLoader } from './CursorLoader';
import { OverlayCard, type CardContent } from './OverlayCard';
import type { OverlayStore } from './overlayStore';
import { StyleMenu } from './StyleMenu';

export function OverlayApp({ store }: { store: OverlayStore }) {
  const { working, review, toast, menu } = useSyncExternalStore(store.subscribe, store.getSnapshot);
  // Until the first words arrive the pointer loader shows the rewrite is under way; from then on the card does.
  const streaming = working?.preview ? { ...working, preview: working.preview } : null;
  // A toast raised meanwhile (a notice, "one at a time") has the card until it closes; what it covered then shows again.
  const card: CardContent | null = toast
    ? { kind: 'toast', toast }
    : review
      ? { kind: 'review', review }
      : streaming
        ? { kind: 'writing', working: streaming }
        : null;
  return (
    <>
      {menu ? <StyleMenu menu={menu} /> : null}
      {working && !streaming ? <CursorLoader working={working} /> : null}
      {card ? <OverlayCard card={card} onDismiss={() => toast && store.dismissToast(toast.id)} /> : null}
    </>
  );
}
