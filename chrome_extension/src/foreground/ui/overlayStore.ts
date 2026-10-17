/**
 * State of the in-page overlay: at most one "working" indicator, one
 * finished rewrite waiting for the user's decision, one toast and one style
 * menu. The working rewrite's live preview, the finished rewrite and the
 * toast share one card on the page (see OverlayCard).
 * A tiny external store so the controller (plain TypeScript) drives React
 * through `useSyncExternalStore` without React owning the job logic.
 */
import type { ErrorPayload } from '../../shared/errors';
import type { AccessOffer, AccessResult, Point } from '../../shared/messages';
import type { StyleId } from '../../shared/styles';

export type { Point };

export interface AnchorRect {
  readonly top: number;
  readonly left: number;
  readonly bottom: number;
  readonly right: number;
}

export type WorkingPhase = 'starting' | 'loading-model' | 'rewriting';

export interface WorkingState {
  readonly style: StyleId;
  readonly phase: WorkingPhase;
  /** 0..1, only while the model is loading. */
  readonly progress?: number;
  /** Where the indicator appears until the pointer moves. */
  readonly origin: Point;
  /** The text being rewritten: the live preview sits beside it. */
  readonly anchor: AnchorRect;
  /** The rewrite so far, while the model writes it (live preview on). */
  readonly preview?: string;
}

export interface ToastAction {
  readonly label: string;
  readonly run: () => void;
}

export type ToastTone = 'success' | 'info' | 'error';

/** A tip set apart under a toast's message and actions, with a button of its own. */
export interface ToastNote {
  readonly text: string;
  readonly action: ToastAction;
}

/**
 * A finished rewrite the user watched being written, shown in full until they
 * decide: Replace puts it in place of their text, Discard leaves the text alone.
 */
export interface ReviewState {
  readonly style: StyleId;
  readonly anchor: AnchorRect;
  readonly text: string;
  /** How long the rewrite took, e.g. "Took 1.2 s". */
  readonly detail: string;
  readonly note?: ToastNote;
  readonly replace: () => void;
  readonly discard: () => void;
}

/** "Text replaced" with an invitation to allow hold and shortcuts: what it is for, and what Allow does. */
export interface SiteAccessOffer extends AccessOffer {
  /** Show Chrome's permission prompt (it resolves once the user has answered). */
  readonly allow: () => Promise<AccessResult>;
}

/** Where the in-place download stands, as the card shows it. */
export type SetupView =
  | { readonly phase: 'checking' }
  | { readonly phase: 'offer'; readonly totalBytes: number; /** The styles that come with it, when the service worker says. */ readonly styles?: readonly StyleId[] }
  | { readonly phase: 'downloading'; readonly verifying: boolean; readonly receivedBytes: number; readonly totalBytes: number; readonly bytesPerSecond: number }
  | { readonly phase: 'done' }
  | { readonly phase: 'failed'; readonly error: ErrorPayload };

/** A rewrite needs the writing model, or a style, that is not on this device yet: the card that downloads it in place. */
export interface SetupState {
  /** What is missing; a style is named for the card's wording. */
  readonly subject: { readonly kind: 'model' } | { readonly kind: 'style'; readonly name: string };
  /** The style the user asked for when this stopped them; once it is installed the card offers to carry on with it. */
  readonly style: StyleId | null;
  readonly anchor: AnchorRect | null;
  readonly view: SetupView;
  /** Start the download (also the retry of one that failed). */
  readonly download: () => void;
  /** Rewrite the selected text in `style`, now that it can: closes the card and starts the rewrite. */
  readonly rewrite: () => void;
  /** Open the toolbar popup on the Performance settings. */
  readonly openSettings: () => void;
  /** Close the card; a download that is running carries on. */
  readonly close: () => void;
}

export interface ToastState {
  readonly id: number;
  readonly tone: ToastTone;
  readonly title: string;
  readonly message?: string;
  readonly actions: readonly ToastAction[];
  readonly note?: ToastNote;
  /** One slim line (the title, the action if any, and a close button) instead of the full card. */
  readonly compact?: boolean;
  /** With `compact`: the toast grows into the offer card, keeping its title and Undo as its first line. */
  readonly offer?: SiteAccessOffer;
  readonly anchor: AnchorRect | null;
  /** Milliseconds before it hides itself; 0 keeps it until dismissed. */
  readonly durationMs: number;
}

/** The styles to pick from, by mouse or keyboard; the controller owns what a pick does. */
export interface StyleMenuState {
  /** Where the hold was: the menu opens beside it. */
  readonly point: Point;
  readonly styles: readonly StyleId[];
  /** The style Enter would pick; the pointer moves it too. */
  readonly active: number;
  readonly pick: (style: StyleId) => void;
  readonly setActive: (index: number) => void;
}

export interface OverlayState {
  readonly working: WorkingState | null;
  readonly review: ReviewState | null;
  readonly toast: ToastState | null;
  readonly menu: StyleMenuState | null;
  /** The download card; it waits behind a toast or a rewrite's own card. */
  readonly setup: SetupState | null;
}

export class OverlayStore {
  private state: OverlayState = { working: null, review: null, toast: null, menu: null, setup: null };
  private readonly listeners = new Set<() => void>();
  private nextToastId = 1;

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): OverlayState => this.state;

  private set(next: OverlayState): void {
    this.state = next;
    for (const listener of this.listeners) listener();
  }

  startWorking(working: WorkingState): void {
    // A new job replaces whatever the previous one left (its rewrite to review, its toast), and the menu it was picked from.
    this.set({ working, review: null, toast: null, menu: null, setup: null });
  }

  updateWorking(patch: Partial<Pick<WorkingState, 'phase' | 'progress' | 'preview'>>): void {
    if (!this.state.working) return;
    this.set({ ...this.state, working: { ...this.state.working, ...patch } });
  }

  /**
   * The job is over. Its outcome, if any, is shown in the same update: the
   * card that previewed the rewrite becomes the toast with the result,
   * rather than closing and a new one opening.
   */
  stopWorking(outcome?: Omit<ToastState, 'id'>): void {
    if (!this.state.working && !outcome) return;
    this.set({ ...this.state, working: null, toast: outcome ? { ...outcome, id: this.nextToastId++ } : this.state.toast });
  }

  /** The job is over and its rewrite waits for the user, in the card that showed it being written. */
  startReview(review: ReviewState): void {
    this.set({ ...this.state, working: null, review });
  }

  /** The user decided. What came of it, if anything to say, takes the same card in the same update. */
  endReview(outcome?: Omit<ToastState, 'id'>): void {
    if (!this.state.review && !outcome) return;
    this.set({ ...this.state, review: null, toast: outcome ? { ...outcome, id: this.nextToastId++ } : this.state.toast });
  }

  showToast(toast: Omit<ToastState, 'id'>): number {
    const id = this.nextToastId++;
    this.set({ ...this.state, toast: { ...toast, id } });
    return id;
  }

  dismissToast(id?: number): void {
    if (!this.state.toast || (id !== undefined && this.state.toast.id !== id)) return;
    this.set({ ...this.state, toast: null });
  }

  /** Opening the menu clears a toast still showing: the menu is what the user is working with now. */
  openMenu(menu: StyleMenuState): void {
    this.set({ ...this.state, toast: null, menu });
  }

  showSetup(setup: SetupState): void {
    this.set({ ...this.state, setup });
  }

  updateSetup(view: SetupView): void {
    if (this.state.setup) this.set({ ...this.state, setup: { ...this.state.setup, view } });
  }

  closeSetup(): void {
    if (this.state.setup) this.set({ ...this.state, setup: null });
  }

  setMenuActive(active: number): void {
    if (this.state.menu) this.set({ ...this.state, menu: { ...this.state.menu, active } });
  }

  closeMenu(): void {
    if (this.state.menu) this.set({ ...this.state, menu: null });
  }
}
