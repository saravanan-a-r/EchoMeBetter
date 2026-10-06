/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { workingLabel } from '../../../foreground/ui/CursorLoader';
import { isBusyCursorOn, setBusyCursor } from '../../../foreground/ui/busyCursor';
import { mountOverlay, OVERLAY_TAG } from '../../../foreground/ui/mountOverlay';
import { OverlayApp } from '../../../foreground/ui/OverlayApp';
import { OverlayStore } from '../../../foreground/ui/overlayStore';
import { toastPosition } from '../../../foreground/ui/Toast';

describe('overlay', () => {
  test('the loader explains each phase', () => {
    const base = { style: 'concise' as const, origin: { x: 0, y: 0 } };
    expect(workingLabel({ ...base, phase: 'starting' })).toBe('Getting ready…');
    expect(workingLabel({ ...base, phase: 'loading-model', progress: 0.42 })).toBe('Waking up the writing model · 42%');
    expect(workingLabel({ ...base, phase: 'rewriting' })).toBe('Rewriting · Concise');
  });

  test('renders the loader and a toast whose action runs and closes it', () => {
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    act(() => store.startWorking({ style: 'friendly', phase: 'rewriting', origin: { x: 10, y: 10 } }));
    expect(screen.getByRole('status', { name: 'Rewriting · Friendly' })).toBeInTheDocument();

    const undo = jest.fn();
    act(() => {
      store.stopWorking();
      store.showToast({ tone: 'success', title: 'Rewritten · Friendly', actions: [{ label: 'Undo', run: undo }], anchor: null, durationMs: 0 });
    });
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(undo).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Rewritten · Friendly')).not.toBeInTheDocument();
  });

  test('toasts dismiss themselves after their duration', () => {
    jest.useFakeTimers();
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    act(() => void store.showToast({ tone: 'info', title: 'Hello', actions: [], anchor: null, durationMs: 4000 }));
    act(() => void jest.advanceTimersByTime(3999));
    expect(screen.getByText('Hello')).toBeInTheDocument();
    act(() => void jest.advanceTimersByTime(1));
    expect(screen.queryByText('Hello')).not.toBeInTheDocument();
    jest.useRealTimers();
  });

  test('toasts sit below the text when there is room, above it otherwise', () => {
    const viewport = { width: 1000, height: 800 };
    expect(toastPosition({ top: 100, left: 50, bottom: 140, right: 400 }, viewport)).toEqual({ left: 50, top: 150 });
    expect(toastPosition({ top: 700, left: 50, bottom: 760, right: 400 }, viewport)).toEqual({ left: 50, bottom: 110 });
    expect(toastPosition({ top: 100, left: 950, bottom: 140, right: 990 }, viewport).left).toBe(630);
  });

  test('mounting adds one isolated host; unmounting removes it', () => {
    const overlay = mountOverlay(document);
    expect(document.querySelectorAll(OVERLAY_TAG)).toHaveLength(1);
    expect(overlay.host.shadowRoot).not.toBeNull();
    overlay.unmount();
    expect(document.querySelectorAll(OVERLAY_TAG)).toHaveLength(0);
  });

  test('host styles are !important so page rules cannot hide it, and overlay events stay inside', () => {
    const overlay = mountOverlay(document);
    expect(overlay.host.getAttribute('popover')).toBe('manual');
    expect(overlay.host.style.getPropertyPriority('display')).toBe('important');
    expect(overlay.host.style.getPropertyValue('pointer-events')).toBe('none');

    const pageSaw = jest.fn();
    document.addEventListener('click', pageSaw);
    overlay.host.shadowRoot!.querySelector('.echomebetter-root')!.dispatchEvent(new MouseEvent('click', { bubbles: true, composed: true }));
    expect(pageSaw).not.toHaveBeenCalled();
    document.removeEventListener('click', pageSaw);
    overlay.unmount();
  });

  test('the busy pointer leaves no trace on the page afterwards', () => {
    const before = document.documentElement.outerHTML;
    setBusyCursor(document, true);
    expect(isBusyCursorOn(document)).toBe(true);
    setBusyCursor(document, false);
    expect(document.documentElement.outerHTML).toBe(before);
  });
});
