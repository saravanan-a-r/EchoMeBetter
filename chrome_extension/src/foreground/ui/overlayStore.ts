/**
 * State of the in-page overlay: at most one "working" indicator and one toast.
 * A tiny external store so the controller (plain TypeScript) drives React
 * through `useSyncExternalStore` without React owning the job logic.
 */
import type { StyleId } from '../../shared/styles';

export interface Point {
  readonly x: number;
  readonly y: number;
}

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
}

export interface ToastAction {
  readonly label: string;
  readonly run: () => void;
}

export type ToastTone = 'success' | 'info' | 'error';

export interface ToastState {
  readonly id: number;
  readonly tone: ToastTone;
  readonly title: string;
  readonly message?: string;
  readonly actions: readonly ToastAction[];
  readonly anchor: AnchorRect | null;
  /** Milliseconds before it hides itself; 0 keeps it until dismissed. */
  readonly durationMs: number;
}

export interface OverlayState {
  readonly working: WorkingState | null;
  readonly toast: ToastState | null;
}

export class OverlayStore {
  private state: OverlayState = { working: null, toast: null };
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
    // A new job replaces any toast from the previous one.
    this.set({ working, toast: null });
  }

  updateWorking(patch: Partial<Pick<WorkingState, 'phase' | 'progress'>>): void {
    if (!this.state.working) return;
    this.set({ ...this.state, working: { ...this.state.working, ...patch } });
  }

  stopWorking(): void {
    if (this.state.working) this.set({ ...this.state, working: null });
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
}
