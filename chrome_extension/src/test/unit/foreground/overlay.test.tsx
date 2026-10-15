/** @jest-environment jsdom */
import { describe, expect, jest, test } from '@jest/globals';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { workingLabel } from '../../../foreground/ui/CursorLoader';
import { isBusyCursorOn, setBusyCursor } from '../../../foreground/ui/busyCursor';
import { mountOverlay, OVERLAY_TAG } from '../../../foreground/ui/mountOverlay';
import { OverlayApp } from '../../../foreground/ui/OverlayApp';
import { OverlayStore } from '../../../foreground/ui/overlayStore';
import { menuPosition } from '../../../foreground/ui/StyleMenu';
import { PREVIEW_MAX_TEXT_HEIGHT, previewStyle, previewWidthRange, toastPosition } from '../../../foreground/ui/OverlayCard';

describe('overlay', () => {
  test('the loader explains each phase', () => {
    const base = { style: 'concise' as const, origin: { x: 0, y: 0 }, anchor: { top: 0, left: 0, bottom: 0, right: 0 } };
    expect(workingLabel({ ...base, phase: 'starting' })).toBe('Getting ready…');
    expect(workingLabel({ ...base, phase: 'loading-model', progress: 0.42 })).toBe('Waking up the writing model · 42%');
    expect(workingLabel({ ...base, phase: 'rewriting' })).toBe('Rewriting · Concise');
  });

  test('renders the loader and a toast whose action runs and closes it', () => {
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    act(() => store.startWorking({ style: 'friendly', phase: 'rewriting', origin: { x: 10, y: 10 }, anchor: { top: 0, left: 0, bottom: 20, right: 100 } }));
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

  test('one element at a time: the loader until the first words, then the card, which keeps the finished rewrite and then says what came of it, in place', () => {
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    act(() => store.startWorking({ style: 'friendly', phase: 'rewriting', origin: { x: 10, y: 10 }, anchor: { top: 0, left: 0, bottom: 20, right: 100 } }));
    expect(screen.getByRole('status', { name: 'Rewriting · Friendly' })).toBeInTheDocument();
    expect(screen.getAllByRole('status')).toHaveLength(1);

    // The first words: the loader gives way to the card, which shows them (to the eye; screen readers hear its title).
    act(() => store.updateWorking({ preview: 'Hey there' }));
    expect(screen.queryByRole('status', { name: 'Rewriting · Friendly' })).not.toBeInTheDocument();
    const [card] = screen.getAllByRole('status');
    expect(screen.getAllByRole('status')).toHaveLength(1);
    expect(card).toHaveTextContent('Rewriting · Friendly');
    expect(screen.getByText('Hey there').closest('[aria-hidden="true"]')).not.toBeNull();
    expect(card).toHaveTextContent('Esc');

    act(() => store.updateWorking({ preview: 'Hey there, could you' }));
    expect(screen.getAllByRole('status')[0]).toBe(card);

    // Written: the same card keeps the text (now for screen readers too, without the caret), in the same scroller.
    const scroller = screen.getByText(/Hey there, could you/).closest('div')!;
    const caret = '[class*="animate-echo-wave"]';
    expect(card!.querySelector(caret)).not.toBeNull();
    const replace = jest.fn();
    const discard = jest.fn();
    act(() =>
      store.startReview({ style: 'friendly', anchor: { top: 0, left: 0, bottom: 20, right: 100 }, text: 'Hey there, could you send it?', detail: 'Took 1.2 s', replace, discard }),
    );
    expect(screen.getAllByRole('status')).toEqual([card]);
    expect(card).toHaveTextContent('Rewrite ready · Friendly');
    expect(card).toHaveTextContent('Took 1.2 s');
    expect(screen.getByText('Hey there, could you send it?').closest('div')).toBe(scroller);
    expect(scroller).not.toHaveAttribute('aria-hidden');
    expect(card!.querySelector(caret)).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Discard' }));
    expect(discard).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole('button', { name: 'Replace' }));
    expect(replace).toHaveBeenCalledTimes(1);

    // Replaced: the same card says so, with Undo.
    const undo = jest.fn();
    act(() => store.endReview({ tone: 'success', title: 'Replaced · Friendly', actions: [{ label: 'Undo', run: undo }], anchor: null, durationMs: 0 }));
    expect(screen.getAllByRole('status')).toEqual([card]);
    expect(card).toHaveTextContent('Replaced · Friendly');
    expect(screen.queryByText(/Hey there/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(undo).toHaveBeenCalledTimes(1);
    expect(card).not.toBeInTheDocument();
  });

  test('a toast raised while the preview runs has the card until it closes; the preview then carries on', () => {
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    act(() => {
      store.startWorking({ style: 'concise', phase: 'rewriting', origin: { x: 10, y: 10 }, anchor: { top: 0, left: 0, bottom: 20, right: 100 } });
      store.updateWorking({ preview: 'Send it' });
    });
    act(() => {
      store.showToast({ tone: 'info', title: 'One at a time', actions: [], anchor: null, durationMs: 0 });
    });
    expect(screen.getByText('One at a time')).toBeInTheDocument();
    expect(screen.queryByText('Send it')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(screen.getByText('Send it')).toBeInTheDocument();
  });

  test("a toast's note shows under its actions, and its button runs and closes the toast", () => {
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    const keepAwake = jest.fn();
    act(() => {
      store.showToast({
        tone: 'success',
        title: 'Rewritten · Concise',
        actions: [{ label: 'Undo', run: jest.fn() }],
        note: { text: 'Waking up took 3.4 s.', action: { label: 'Keep it awake longer', run: keepAwake } },
        anchor: null,
        durationMs: 0,
      });
    });
    expect(screen.getByText('Waking up took 3.4 s.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Keep it awake longer' }));
    expect(keepAwake).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('Rewritten · Concise')).not.toBeInTheDocument();
  });

  test('the style menu lists every style with its hint; clicking one picks it, hovering makes it the active one', () => {
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    const pick = jest.fn();
    act(() => {
      store.openMenu({ point: { x: 20, y: 20 }, styles: ['professional', 'grammar', 'friendly', 'concise', 'elaborate'], active: 0, pick, setActive: (index) => store.setMenuActive(index) });
    });
    const menu = screen.getByRole('menu', { name: 'Rewrite as' });
    const items = screen.getAllByRole('menuitem');
    expect(items.map((item) => item.textContent)).toEqual([
      'ProfessionalPolished and workplace-ready1',
      'GrammarFix grammar and spelling only2',
      'FriendlyWarm and approachable3',
      'ConciseShorter, same meaning4',
      'ElaborateFuller, with more detail5',
    ]);
    expect(menu).toBeInTheDocument();
    fireEvent.mouseEnter(items[3]!);
    expect(screen.getAllByRole('menuitem')[3]).toHaveAttribute('data-active', 'true');
    fireEvent.click(items[1]!);
    expect(pick).toHaveBeenCalledWith('grammar');
  });

  test('the menu opens beside the point, and on its other side near the edges of the window', () => {
    const size = { width: 272, height: 300 };
    const viewport = { width: 1000, height: 800 };
    expect(menuPosition({ x: 100, y: 100 }, size, viewport)).toEqual({ left: 110, top: 114 });
    expect(menuPosition({ x: 900, y: 700 }, size, viewport)).toEqual({ left: 900 - 10 - 272, top: 700 - 14 - 300 });
    // A window too small for either side: kept on screen.
    expect(menuPosition({ x: 150, y: 150 }, size, { width: 300, height: 320 })).toEqual({ left: 8, top: 8 });
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

  test('the preview is as wide as its text wants, from a toast up to 70% of the window, and never off screen', () => {
    expect(previewWidthRange({ width: 1000 })).toEqual({ min: 360, max: 700 });
    // A narrow window: the toast's own limit wins over 70%.
    expect(previewWidthRange({ width: 400 })).toEqual({ min: 360, max: 360 });
    expect(previewWidthRange({ width: 300 })).toEqual({ min: 280, max: 280 });

    const style = previewStyle({ top: 100, left: 50, bottom: 140, right: 400 }, { width: 1000, height: 800 });
    expect(style).toMatchObject({ left: 50, top: 150, width: 'max-content', minWidth: 360, maxWidth: 700 });
    // Room to the right is 940px (to the 10px margin): a card up to that wide needs no shift.
    expect(style.translate).toBe('min(0px, calc(940px - 100%))');
    // Near the right edge the toast's own clamp puts it at 630; the card then shifts left by whatever it overflows.
    expect(previewStyle({ top: 100, left: 950, bottom: 140, right: 990 }, { width: 1000, height: 800 }).translate).toBe('min(0px, calc(360px - 100%))');
  });

  test('the preview scrolls past its height, follows the newest text, and lets go once the reader scrolls up', () => {
    const store = new OverlayStore();
    render(<OverlayApp store={store} />);
    act(() => store.startWorking({ style: 'concise', phase: 'rewriting', origin: { x: 0, y: 0 }, anchor: { top: 0, left: 0, bottom: 20, right: 100 } }));
    act(() => store.updateWorking({ preview: 'First line' }));
    const scroller = screen.getByText(/First line/).closest('[aria-hidden="true"]') as HTMLElement;
    expect(scroller.style.maxHeight).toBe(`${PREVIEW_MAX_TEXT_HEIGHT}px`);
    expect(scroller.className).toContain('overflow-y-auto');

    // jsdom lays nothing out: stand in for a 400px-tall text in a 120px window.
    let height = 400;
    Object.defineProperties(scroller, {
      scrollHeight: { configurable: true, get: () => height },
      clientHeight: { configurable: true, value: PREVIEW_MAX_TEXT_HEIGHT },
    });
    act(() => store.updateWorking({ preview: 'First line\nand more' }));
    expect(scroller.scrollTop).toBe(400);

    // Scrolled up to read: new text no longer pulls the view down.
    scroller.scrollTop = 50;
    fireEvent.scroll(scroller);
    height = 440;
    act(() => store.updateWorking({ preview: 'First line\nand more\nand more again' }));
    expect(scroller.scrollTop).toBe(50);

    // Back at the end: it follows again.
    scroller.scrollTop = height - PREVIEW_MAX_TEXT_HEIGHT;
    fireEvent.scroll(scroller);
    height = 480;
    act(() => store.updateWorking({ preview: 'First line\nand more\nand more again\nlast' }));
    expect(scroller.scrollTop).toBe(480);
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
