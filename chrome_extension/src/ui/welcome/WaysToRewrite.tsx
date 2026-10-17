/**
 * "Three ways in. One result.": the welcome page's guide to starting a
 * rewrite, one tab per way (right-click, press and hold, keyboard). Each tab
 * plays a drawn demo of that way next to its steps, ending the same way every
 * time: the rewrite is written in a card beside the text, and only Replace
 * changes it.
 *
 * The demos loop only while they are on screen, and hold still on their
 * result for anyone who prefers reduced motion.
 */
import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { shortcutKeys, shortcutLabel, type KeyPlatform } from '../../shared/shortcuts';
import { STYLES, styleLabel, type StyleId } from '../../shared/styles';
import { Logo } from '../brand/Logo';
import { StyleGlyph } from '../brand/StyleGlyph';

export type Way = 'right-click' | 'hold' | 'keys';

const WAYS: Record<Way, { tab: string; site: string; text: string; style: StyleId; result: string }> = {
  'right-click': {
    tab: 'Right-click',
    site: 'mail.example.com',
    text: 'gonna be 10 min late, traffic is crazy rn',
    style: 'professional',
    result: "I'm running about 10 minutes late due to heavy traffic. Sorry for the wait!",
  },
  hold: {
    tab: 'Press and hold',
    site: 'chat.example.com',
    text: 'we was planing to send it yesterday but their was a problem',
    style: 'grammar',
    result: 'We were planning to send it yesterday, but there was a problem.',
  },
  keys: {
    tab: 'Keyboard',
    site: 'docs.example.com',
    text: 'We are writing to let you know that the meeting that was scheduled for Monday has now been moved to Wednesday, at the same time as before.',
    style: 'concise',
    result: "Monday's meeting moves to Wednesday, same time.",
  },
};

/**
 * How much slower than real life the demos play, so each move can be followed.
 * The press and hold ring alone keeps its real second.
 */
const PACE = 1.8;

const TITLES = ['', 'How it works', 'Two ways in. One result.', 'Three ways in. One result.'];

/** `platform`: the keyboard tab, while shortcuts are on. `hold`: the press and hold tab, while it is on. */
export function WaysToRewrite({ platform, hold }: { platform?: KeyPlatform; hold: boolean }) {
  const ways: Way[] = ['right-click', ...(hold ? (['hold'] as const) : []), ...(platform ? (['keys'] as const) : [])];
  const [chosen, setChosen] = useState<Way>('right-click');
  const way = ways.includes(chosen) ? chosen : 'right-click';
  const tabs = useRef<Partial<Record<Way, HTMLButtonElement | null>>>({});
  const [step, setStep] = useState(0);

  const onKeyDown = (event: KeyboardEvent) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    const next = ways[(ways.indexOf(way) + step + ways.length) % ways.length]!;
    setChosen(next);
    tabs.current[next]?.focus();
  };

  return (
    <section aria-labelledby="ways-title" className="_echo_$_rounded-3xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_p-4 _echo_$_shadow-card sm:_echo_$_p-6">
      <div className="_echo_$_flex _echo_$_flex-wrap _echo_$_items-start _echo_$_justify-between _echo_$_gap-4">
        <div className="_echo_$_max-w-xl">
          <h2 id="ways-title" className="_echo_$_text-2xl _echo_$_font-bold">
            {TITLES[ways.length]}
          </h2>
          <p className="_echo_$_mt-1 _echo_$_text-sm _echo_$_text-fg-muted">
            {ways.length > 1 ? 'Whichever you use, the' : 'The'} rewrite appears beside your text, and your text changes only when you choose Replace.
          </p>
        </div>
      {ways.length > 1 ? (
        <div role="tablist" aria-label="Ways to rewrite" onKeyDown={onKeyDown} className="_echo_$_flex _echo_$_w-full _echo_$_gap-1 _echo_$_rounded-xl _echo_$_bg-muted _echo_$_p-1 sm:_echo_$_w-auto">
          {ways.map((id) => (
            <button
              key={id}
              ref={(element) => {
                tabs.current[id] = element;
              }}
              type="button"
              role="tab"
              id={`way-tab-${id}`}
              aria-selected={id === way}
              aria-controls="way-panel"
              tabIndex={id === way ? 0 : -1}
              onClick={() => setChosen(id)}
              className={`_echo_$_flex-1 _echo_$_cursor-pointer _echo_$_whitespace-nowrap _echo_$_rounded-lg _echo_$_border-0 _echo_$_px-1.5 _echo_$_py-1.5 _echo_$_text-[13px] _echo_$_font-semibold sm:_echo_$_flex-none sm:_echo_$_text-sm sm:_echo_$_px-4 _echo_$_transition focus-visible:_echo_$_outline focus-visible:_echo_$_outline-2 focus-visible:_echo_$_outline-focus ${
                id === way ? '_echo_$_bg-segment-active _echo_$_text-accent-fg-strong _echo_$_shadow-sm' : '_echo_$_bg-transparent _echo_$_text-fg-soft hover:_echo_$_bg-segment-hover'
              }`}
            >
              {WAYS[id].tab}
            </button>
          ))}
        </div>
      ) : null}
      </div>
      <div
        id="way-panel"
        role={ways.length > 1 ? 'tabpanel' : undefined}
        aria-labelledby={ways.length > 1 ? `way-tab-${way}` : undefined}
        className="_echo_$_mt-5 _echo_$_grid _echo_$_grid-cols-1 _echo_$_items-start _echo_$_gap-6 md:_echo_$_grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]"
      >
        <div className="_echo_$_rounded-2xl _echo_$_border _echo_$_border-line _echo_$_bg-canvas _echo_$_p-4">
          <WayDemo key={way} way={way} platform={platform ?? 'other'} onStep={setStep} />
        </div>
        <WaySteps way={way} platform={platform} active={step} />
      </div>
    </section>
  );
}

const DECIDE: [string, string] = ['Replace ↵ or Discard Esc', 'Review the rewrite first. If you change your mind after replacing, choose Undo.'];

/** The steps of one way; `active`, the one its demo is showing, is lit. */
function WaySteps({ way, platform, active }: { way: Way; platform?: KeyPlatform; active: number }) {
  const steps: [string, string][] =
    way === 'right-click'
      ? [
          ['Select your text', 'A word, a sentence or the whole message, in any text box.'],
          ['Right-click › EchoMeBetter', 'The writing styles open in a submenu.'],
          ['Choose a style', 'The rewrite appears beside your text within seconds.'],
          DECIDE,
        ]
      : way === 'hold'
        ? [
            ['Select your text', 'Then rest the pointer on the selection.'],
            ['Press and hold for 1 second', 'Keep the pointer still while the ring fills.'],
            ['Choose a style', 'Click a style, or press 1–5 on your keyboard.'],
            DECIDE,
          ]
        : [['Select your text', 'In any text field.'], ['Press a shortcut', 'Each style has its own:'], DECIDE];
  return (
    <ol className="_echo_$_space-y-1.5">
      {steps.map(([title, body], index) => {
        const lit = index + 1 === active;
        return (
          <li
            key={title}
            aria-current={lit ? 'step' : undefined}
            className={`_echo_$_flex _echo_$_gap-3 _echo_$_rounded-xl _echo_$_p-2.5 _echo_$_transition-colors _echo_$_duration-500 ${lit ? '_echo_$_bg-accent-subtle' : ''}`}
          >
            <span
              className={`_echo_$_flex _echo_$_h-8 _echo_$_w-8 _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-full _echo_$_text-sm _echo_$_font-bold _echo_$_transition-colors _echo_$_duration-500 ${
                lit ? '_echo_$_bg-accent _echo_$_text-on-accent' : '_echo_$_bg-accent-subtle _echo_$_text-accent-fg-strong'
              }`}
            >
              {index + 1}
            </span>
            <div className="_echo_$_min-w-0 _echo_$_flex-1">
              <h3 className="_echo_$_font-semibold">{title}</h3>
              <p className="_echo_$_text-sm _echo_$_text-fg-muted">{body}</p>
              {way === 'keys' && index === 1 && platform ? (
                <ul className="_echo_$_mt-3 _echo_$_space-y-1.5">
                  {STYLES.map((style) => (
                    <li
                      key={style.id}
                      className="_echo_$_flex _echo_$_flex-wrap _echo_$_items-center _echo_$_justify-between _echo_$_gap-x-3 _echo_$_gap-y-0.5 _echo_$_rounded-xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_px-3 _echo_$_py-1.5 _echo_$_text-sm _echo_$_font-medium"
                    >
                      <span className="_echo_$_flex _echo_$_items-center _echo_$_gap-2">
                        <span className="_echo_$_text-accent-fg">
                          <StyleGlyph style={style.id} size={16} />
                        </span>
                        {style.label}
                      </span>
                      <kbd className="_echo_$_shrink-0 _echo_$_whitespace-nowrap _echo_$_font-sans _echo_$_text-fg-secondary">{shortcutLabel(style.id, platform)}</kbd>
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

// ---------- the demo ----------

interface Point {
  readonly x: number;
  readonly y: number;
}

interface Scene {
  readonly rewritten: boolean;
  /** How many words of the text are selected. */
  readonly selected: number;
  readonly cursor: { at: Point; ms: number; down: boolean; shown: boolean };
  readonly presses: readonly { id: number; at: Point }[];
  readonly ring: Point | null;
  /** Chrome's right-click menu, and its EchoMeBetter submenu. */
  readonly context: { at: Point; echo: boolean } | null;
  readonly submenu: { at: Point; active: StyleId | null } | null;
  /** The press and hold style menu. */
  readonly menu: { at: Point; active: StyleId | null } | null;
  readonly card: { phase: 'writing' | 'ready' | 'replaced'; words: number; top: number } | null;
  readonly keys: { caps: readonly string[]; down: number; top: number } | null;
  readonly flash: number;
  /** The step of the list beside the demo it is showing; 0 for none. */
  readonly step: number;
}

const START: Scene = {
  rewritten: false,
  selected: 0,
  cursor: { at: { x: 0, y: 0 }, ms: 0, down: false, shown: false },
  presses: [],
  ring: null,
  context: null,
  submenu: null,
  menu: null,
  card: null,
  keys: null,
  flash: 0,
  step: 0,
};

class Stopped extends Error {}

function prefersReducedMotion(): boolean {
  return typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** A looping, drawn demo of one way to rewrite: a text box, the pointer, the menus and the card. */
function WayDemo({ way, platform, onStep }: { way: Way; platform: KeyPlatform; onStep: (step: number) => void }) {
  const demo = WAYS[way];
  const words = demo.text.split(' ');
  const result = demo.result.split(' ');
  const stage = useRef<HTMLDivElement>(null);
  const [scene, setScene] = useState<Scene>(START);
  useEffect(() => onStep(scene.step), [scene.step, onStep]);

  useEffect(() => {
    const root = stage.current!;
    let run = 0;
    const set = (patch: Partial<Scene> | ((scene: Scene) => Partial<Scene>)) =>
      setScene((scene) => ({ ...scene, ...(typeof patch === 'function' ? patch(scene) : patch) }));
    const box = (selector: string) => {
      const element = root.querySelector<HTMLElement>(selector)!;
      const outer = root.getBoundingClientRect();
      const rect = element.getBoundingClientRect();
      return { left: rect.left - outer.left, top: rect.top - outer.top, width: rect.width, height: rect.height };
    };
    const at = (selector: string, fx = 0.5, fy = 0.5): Point => {
      const rect = box(selector);
      return { x: rect.left + rect.width * fx, y: rect.top + rect.height * fy };
    };
    const windowHeight = () => box('[data-demo="window"]').height;
    const settled = (): Partial<Scene> => ({ ...START, selected: words.length, card: { phase: 'ready', words: result.length, top: windowHeight() + 12 } });

    if (prefersReducedMotion() || typeof IntersectionObserver === 'undefined') {
      if (prefersReducedMotion()) set(settled());
      return;
    }

    async function play(id: number) {
      const wait = (ms: number, pace = PACE) =>
        new Promise<void>((resolve, reject) => setTimeout(() => (id === run ? resolve() : reject(new Stopped())), ms * pace));
      const move = (to: Point, ms = 500, more: Partial<Scene['cursor']> = {}) => set((scene) => ({ cursor: { ...scene.cursor, at: to, ms: ms * PACE, ...more } }));
      const press = (to: Point) => {
        const mark = { id: Math.random(), at: to };
        set((scene) => ({ presses: [...scene.presses, mark] }));
        setTimeout(() => set((scene) => ({ presses: scene.presses.filter((other) => other !== mark) })), 600);
      };
      const click = async (selector: string, fx = 0.5) => {
        move(at(selector, fx), 520);
        await wait(650);
        press(at(selector, fx));
        await wait(200);
      };
      async function select() {
        set({ step: 1 });
        move(at('[data-word="0"]', 0, 0.6), 650);
        await wait(800);
        move(at('[data-word="0"]', 0, 0.6), 0, { down: true });
        for (let index = 0; index < words.length; index++) {
          set({ selected: index + 1 });
          move(at(`[data-word="${index}"]`, 1, 0.6), 85);
          await wait(85);
        }
        set((scene) => ({ cursor: { ...scene.cursor, down: false } }));
        await wait(350);
      }
      async function write() {
        set({ card: { phase: 'writing', words: 0, top: windowHeight() + 12 } });
        for (let count = 1; count <= result.length; count++) {
          set((scene) => ({ card: { ...scene.card!, words: count } }));
          await wait(40, 1); // the rewrite streams in at its own quick pace, not slowed with the rest
        }
        set((scene) => ({ card: { ...scene.card!, phase: 'ready' }, step: scene.step + 1 }));
        await wait(700);
      }
      async function replaced() {
        set((scene) => ({ rewritten: true, selected: 0, flash: scene.flash + 1, card: { ...scene.card!, phase: 'replaced' } }));
        await wait(2300);
        set({ card: null, step: 0 });
        await wait(900);
      }

      set({ ...START, cursor: { at: { x: root.clientWidth * 0.82, y: 250 }, ms: 0, down: false, shown: way !== 'keys' } });
      await wait(600);

      if (way === 'right-click') {
        await select();
        const spot = at(`[data-word="${words.length - 1}"]`, 0.6, 0.6);
        press(spot);
        set({ context: { at: { x: Math.min(spot.x + 2, root.clientWidth - 184), y: spot.y + 4 }, echo: false }, step: 2 });
        await wait(550);
        move(at('[data-demo="echo"]', 0.4), 420);
        await wait(450);
        const menu = box('[data-demo="context"]');
        const row = box('[data-demo="echo"]');
        const right = menu.left + menu.width - 3;
        // Beside the menu as Chrome opens it, on whichever side has room; on a narrow screen, under the row.
        const submenuAt = (menu: { left: number }, row: { top: number; height: number }, right: number): Point =>
          right + 160 <= root.clientWidth
            ? { x: right, y: row.top - 4 }
            : menu.left >= 160
              ? { x: menu.left - 160, y: row.top - 4 }
              : { x: Math.max(0, root.clientWidth - 160), y: row.top + row.height + 4 };
        set((scene) => ({
          context: { ...scene.context!, echo: true },
          submenu: { at: submenuAt(menu, row, right), active: null },
        }));
        await wait(300);
        move(at('[data-demo="echo"]', 0.98), 220);
        await wait(240);
        const target = `[data-demo="submenu"] [data-style="${demo.style}"]`;
        move(at(target, 0.4), 380);
        await wait(420);
        set((scene) => ({ submenu: { ...scene.submenu!, active: demo.style }, step: 3 }));
        await wait(250);
        press(at(target, 0.4));
        await wait(150);
        set((scene) => ({ context: null, submenu: null, cursor: { ...scene.cursor, shown: false } }));
      } else if (way === 'hold') {
        await select();
        const spot = at(`[data-word="${words.length - 1}"]`, 0.5, 0.6);
        move(spot, 260);
        await wait(320);
        set((scene) => ({ ring: spot, cursor: { ...scene.cursor, down: true }, step: 2 }));
        await wait(1050, 1);
        set({ ring: null, menu: { at: { x: Math.min(spot.x + 10, root.clientWidth - 240), y: spot.y + 14 }, active: 'professional' } });
        await wait(350);
        set((scene) => ({ cursor: { ...scene.cursor, down: false } }));
        await wait(350);
        const target = `[data-demo="menu"] [data-style="${demo.style}"]`;
        move(at(target, 0.35), 450);
        await wait(470);
        set((scene) => ({ menu: { ...scene.menu!, active: demo.style }, step: 3 }));
        await wait(300);
        press(at(target, 0.35));
        await wait(150);
        set((scene) => ({ menu: null, cursor: { ...scene.cursor, shown: false } }));
      } else {
        await wait(300);
        set({ step: 1 });
        for (let index = 0; index < words.length; index++) {
          set({ selected: index + 1 });
          await wait(45);
        }
        await wait(450);
        const caps = shortcutKeys(demo.style, platform);
        set({ keys: { caps, down: 0, top: windowHeight() + 24 }, step: 2 });
        for (let index = 1; index <= caps.length; index++) {
          await wait(index === 1 ? 0 : 220);
          set((scene) => ({ keys: { ...scene.keys!, down: index } }));
        }
        await wait(670);
        set({ keys: null });
        await write();
        set({ keys: { caps: ['↵ Enter'], down: 0, top: box('[data-demo="card"]').top + box('[data-demo="card"]').height + 16 } });
        await wait(450);
        set((scene) => ({ keys: { ...scene.keys!, down: 1 } }));
        await wait(300);
        set({ keys: null });
        await replaced();
        return;
      }
      await write();
      set((scene) => ({ cursor: { ...scene.cursor, shown: true } }));
      await click('[data-demo="replace"]');
      await replaced();
    }

    async function loop(id: number) {
      try {
        while (id === run) await play(id);
      } catch (error) {
        if (!(error instanceof Stopped)) throw error;
      }
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        run += 1;
        if (entry?.isIntersecting) void loop(run);
        else setScene(START);
      },
      { threshold: 0.3 },
    );
    observer.observe(root);
    return () => {
      run += 1;
      observer.disconnect();
    };
    // The demo restarts (by its key) when the tab changes; nothing else feeds it.
  }, []);

  const label = styleLabel(demo.style);
  const shown = scene.rewritten ? result : words;
  return (
    <div ref={stage} aria-hidden="true" className="_echo_$_relative _echo_$_min-h-[300px] _echo_$_select-none _echo_$_font-sans">
      <div data-demo="window" className="_echo_$_overflow-hidden _echo_$_rounded-2xl _echo_$_border _echo_$_border-line _echo_$_bg-surface _echo_$_shadow-card">
        <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-1.5 _echo_$_border-b _echo_$_border-line _echo_$_px-3 _echo_$_py-2.5 _echo_$_text-xs _echo_$_text-fg-muted">
          <i className="_echo_$_h-2 _echo_$_w-2 _echo_$_rounded-full _echo_$_bg-line-strong" />
          <i className="_echo_$_h-2 _echo_$_w-2 _echo_$_rounded-full _echo_$_bg-line-strong" />
          <i className="_echo_$_h-2 _echo_$_w-2 _echo_$_rounded-full _echo_$_bg-line-strong" />
          <span className="_echo_$_ml-2">{demo.site}</span>
        </div>
        <p key={scene.flash} className={`_echo_$_m-0 _echo_$_min-h-[84px] _echo_$_px-4 _echo_$_pb-5 _echo_$_pt-3.5 _echo_$_text-base _echo_$_leading-7 _echo_$_text-fg ${scene.flash ? '_echo_$_animate-echo-flash' : ''}`}>
          {shown.map((word, index) => (
            // The space after a word is selected with the next one, as a browser selects.
            <span key={index} className={index + 1 < scene.selected ? '_echo_$_bg-accent/20' : ''}>
              <span data-word={index} className={index + 1 === scene.selected ? '_echo_$_bg-accent/20' : ''}>
                {word}
              </span>{' '}
            </span>
          ))}
        </p>
      </div>

      {scene.context ? (
        <div
          data-demo="context"
          style={{ left: scene.context.at.x, top: scene.context.at.y }}
          className="_echo_$_absolute _echo_$_z-20 _echo_$_w-44 _echo_$_rounded-lg _echo_$_border _echo_$_border-line-strong _echo_$_bg-surface-raised _echo_$_p-1 _echo_$_text-[13px] _echo_$_text-fg-secondary _echo_$_shadow-float"
        >
          {['Cut', 'Copy', 'Paste'].map((item) => (
            <div key={item} className="_echo_$_px-2.5 _echo_$_py-1">
              {item}
            </div>
          ))}
          <div className="_echo_$_my-1 _echo_$_h-px _echo_$_bg-line" />
          <div
            data-demo="echo"
            className={`_echo_$_flex _echo_$_items-center _echo_$_gap-2 _echo_$_rounded _echo_$_px-2.5 _echo_$_py-1 ${
              scene.context.echo ? '_echo_$_bg-accent-subtle _echo_$_font-medium _echo_$_text-accent-fg-strong' : ''
            }`}
          >
            <Logo size={15} /> EchoMeBetter<span className="_echo_$_ml-auto">›</span>
          </div>
          <div className="_echo_$_my-1 _echo_$_h-px _echo_$_bg-line" />
          <div className="_echo_$_px-2.5 _echo_$_py-1">Inspect</div>
        </div>
      ) : null}

      {scene.submenu ? (
        <div
          data-demo="submenu"
          style={{ left: scene.submenu.at.x, top: scene.submenu.at.y }}
          className="_echo_$_absolute _echo_$_z-20 _echo_$_w-[160px] _echo_$_rounded-lg _echo_$_border _echo_$_border-line-strong _echo_$_bg-surface-raised _echo_$_p-1 _echo_$_text-[13px] _echo_$_text-fg-secondary _echo_$_shadow-float"
        >
          {STYLES.map((style) => (
            <div
              key={style.id}
              data-style={style.id}
              className={`_echo_$_rounded _echo_$_px-2.5 _echo_$_py-1 ${style.id === scene.submenu!.active ? '_echo_$_bg-accent _echo_$_text-on-accent' : ''}`}
            >
              {style.label}
            </div>
          ))}
        </div>
      ) : null}

      {scene.menu ? (
        <div
          data-demo="menu"
          style={{ left: scene.menu.at.x, top: scene.menu.at.y }}
          className="_echo_$_theme-inverse _echo_$_absolute _echo_$_z-20 _echo_$_w-[236px] _echo_$_animate-echo-pop-in _echo_$_rounded-2xl _echo_$_bg-surface _echo_$_p-1.5 _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15"
        >
          <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-1.5 _echo_$_px-2 _echo_$_pb-1 _echo_$_pt-0.5 _echo_$_text-[11px] _echo_$_font-semibold _echo_$_uppercase _echo_$_tracking-wider _echo_$_text-fg-muted">
            <Logo size={14} /> Rewrite as
          </div>
          {STYLES.map((style) => {
            const active = style.id === scene.menu!.active;
            return (
              <div key={style.id} data-style={style.id} className={`_echo_$_flex _echo_$_items-center _echo_$_gap-2.5 _echo_$_rounded-xl _echo_$_px-2 _echo_$_py-1 _echo_$_text-[14px] _echo_$_font-semibold ${active ? '_echo_$_bg-sheen/10' : ''}`}>
                <span
                  className={`_echo_$_flex _echo_$_h-7 _echo_$_w-7 _echo_$_items-center _echo_$_justify-center _echo_$_rounded-lg ${
                    active ? '_echo_$_bg-accent _echo_$_text-on-accent' : '_echo_$_bg-sheen/10 _echo_$_text-accent'
                  }`}
                >
                  <StyleGlyph style={style.id} size={16} />
                </span>
                {style.label}
              </div>
            );
          })}
        </div>
      ) : null}

      {scene.card && scene.card.phase === 'replaced' ? (
        <div
          data-demo="card"
          style={{ top: scene.card.top }}
          className="_echo_$_theme-inverse _echo_$_absolute _echo_$_left-0 _echo_$_z-10 _echo_$_flex _echo_$_w-max _echo_$_max-w-full _echo_$_animate-echo-pop-in _echo_$_items-stretch _echo_$_overflow-hidden _echo_$_rounded-xl _echo_$_bg-surface _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15"
        >
          <span className="_echo_$_w-1.5 _echo_$_shrink-0 _echo_$_bg-success" />
          <div className="_echo_$_flex _echo_$_items-center _echo_$_gap-3 _echo_$_py-2 _echo_$_pl-3 _echo_$_pr-2">
            <span className="_echo_$_flex _echo_$_h-7 _echo_$_shrink-0 _echo_$_items-center">
              <Logo size={20} />
            </span>
            <span className="_echo_$_flex _echo_$_h-7 _echo_$_items-center _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5">Text replaced</span>
            <span className="_echo_$_flex _echo_$_h-7 _echo_$_items-center _echo_$_rounded-md _echo_$_px-2.5 _echo_$_text-[14px] _echo_$_font-semibold _echo_$_leading-5 _echo_$_text-accent">Undo</span>
            <span aria-hidden="true" className="_echo_$_flex _echo_$_h-7 _echo_$_w-7 _echo_$_items-center _echo_$_justify-center _echo_$_text-fg-soft">
              <svg width="14" height="14" viewBox="0 0 14 14">
                <path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
              </svg>
            </span>
          </div>
        </div>
      ) : null}

      {scene.card && scene.card.phase !== 'replaced' ? (
        <div
          data-demo="card"
          style={{ top: scene.card.top }}
          className="_echo_$_theme-inverse _echo_$_absolute _echo_$_left-0 _echo_$_z-10 _echo_$_flex _echo_$_w-full _echo_$_max-w-[400px] _echo_$_animate-echo-pop-in _echo_$_overflow-hidden _echo_$_rounded-2xl _echo_$_bg-surface _echo_$_text-fg _echo_$_shadow-float _echo_$_ring-1 _echo_$_ring-sheen/15"
        >
          <span className="_echo_$_w-1.5 _echo_$_shrink-0 _echo_$_bg-accent" />
          <div className="_echo_$_flex _echo_$_min-w-0 _echo_$_flex-1 _echo_$_gap-3 _echo_$_p-3.5">
            <span className="_echo_$_mt-0.5 _echo_$_shrink-0">
              <Logo size={24} />
            </span>
            <div className="_echo_$_min-w-0 _echo_$_flex-1">
              <p className="_echo_$_m-0 _echo_$_text-[15px] _echo_$_font-semibold _echo_$_leading-[22px]">
                {scene.card.phase === 'writing' ? 'Rewriting' : 'Rewrite ready'} · {label}
              </p>
              <p className="_echo_$_m-0 _echo_$_mt-1 _echo_$_text-sm _echo_$_leading-6 _echo_$_text-fg-secondary">
                  {result.slice(0, scene.card.words).join(' ')}
                  {scene.card.phase === 'writing' ? (
                    <span className="_echo_$_ml-0.5 _echo_$_inline-block _echo_$_h-[1em] _echo_$_w-0.5 _echo_$_translate-y-[0.15em] _echo_$_animate-echo-wave _echo_$_rounded-full _echo_$_bg-accent" />
                  ) : null}
                </p>
              {scene.card.phase === 'ready' ? (
                <div className="_echo_$_mt-2.5 _echo_$_flex _echo_$_items-center _echo_$_gap-2 _echo_$_text-[13px] _echo_$_font-semibold">
                  <span data-demo="replace" className="_echo_$_rounded-lg _echo_$_bg-accent _echo_$_px-2.5 _echo_$_py-1 _echo_$_text-on-accent">
                    Replace ↵
                  </span>
                  <span className="_echo_$_rounded-lg _echo_$_bg-sheen/10 _echo_$_px-2.5 _echo_$_py-1 _echo_$_ring-1 _echo_$_ring-sheen/25">Discard Esc</span>
                  <span className="_echo_$_ml-auto _echo_$_font-normal _echo_$_text-fg-soft">Took 1.6 s</span>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}

      {scene.keys ? (
        <div style={{ top: scene.keys.top }} className="_echo_$_absolute _echo_$_left-1/2 _echo_$_z-20 _echo_$_flex _echo_$_max-w-full _echo_$_-translate-x-1/2 _echo_$_gap-2 sm:_echo_$_gap-3">
          {scene.keys.caps.map((cap, index) => (
            <kbd
              key={cap}
              className={`_echo_$_flex _echo_$_h-11 _echo_$_min-w-[44px] _echo_$_shrink-0 _echo_$_items-center _echo_$_justify-center _echo_$_whitespace-nowrap _echo_$_rounded-xl _echo_$_border _echo_$_px-3 _echo_$_font-sans _echo_$_text-[16px] _echo_$_font-semibold sm:_echo_$_h-12 sm:_echo_$_min-w-[48px] sm:_echo_$_px-3.5 sm:_echo_$_text-[19px] _echo_$_shadow-card _echo_$_transition-all _echo_$_duration-100 ${
                index < scene.keys!.down
                  ? '_echo_$_translate-y-0.5 _echo_$_border-b _echo_$_border-accent _echo_$_bg-accent _echo_$_text-on-accent'
                  : '_echo_$_border-b-[3px] _echo_$_border-line-strong _echo_$_bg-surface _echo_$_text-fg'
              }`}
            >
              {cap}
            </kbd>
          ))}
        </div>
      ) : null}

      {scene.ring ? (
        <svg width="44" height="44" viewBox="0 0 40 40" style={{ left: scene.ring.x - 22, top: scene.ring.y - 22 }} className="_echo_$_pointer-events-none _echo_$_absolute _echo_$_z-30">
          <circle cx="20" cy="20" r="16" fill="none" strokeWidth="3" className="_echo_$_stroke-accent/20" />
          <circle
            cx="20"
            cy="20"
            r="16"
            fill="none"
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray="100.5"
            strokeDashoffset="100.5"
            transform="rotate(-90 20 20)"
            className="_echo_$_animate-echo-hold-fill _echo_$_stroke-accent"
          />
        </svg>
      ) : null}

      {scene.presses.map((mark) => (
        <span
          key={mark.id}
          style={{ left: mark.at.x - 15, top: mark.at.y - 15 }}
          className="_echo_$_pointer-events-none _echo_$_absolute _echo_$_z-30 _echo_$_h-[30px] _echo_$_w-[30px] _echo_$_animate-echo-press _echo_$_rounded-full _echo_$_border-2 _echo_$_border-accent"
        />
      ))}

      <svg
        width="20"
        height="24"
        viewBox="0 0 18 22"
        style={{
          transform: `translate(${scene.cursor.at.x - 2}px, ${scene.cursor.at.y - 2}px) scale(${scene.cursor.down ? 0.86 : 1})`,
          transitionDuration: `${scene.cursor.ms}ms`,
        }}
        className={`_echo_$_pointer-events-none _echo_$_absolute _echo_$_left-0 _echo_$_top-0 _echo_$_z-40 _echo_$_origin-top-left _echo_$_transition-transform _echo_$_ease-[cubic-bezier(0.45,0,0.2,1)] ${
          scene.cursor.shown ? '' : '_echo_$_hidden'
        }`}
      >
        <path d="M1.5 1.5v16.2l4.1-3.9 2.7 6.2 3-1.3-2.7-6.1h5.8z" strokeWidth="1.4" strokeLinejoin="round" className="_echo_$_fill-fg _echo_$_stroke-canvas" />
      </svg>
    </div>
  );
}
