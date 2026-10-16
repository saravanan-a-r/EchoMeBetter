/**
 * End-to-end check of the built extension in a real Chromium (Playwright).
 *
 *   npm run build && npm run test:e2e
 *
 * The model must be served at the URL in model.source.json first (any
 * static server with CORS and HTTP range support); the run downloads it
 * from there exactly as a user would.
 *
 * Loads dist/ as an unpacked extension in a fresh profile, then:
 *   1. a rewrite before the download is refused with "download the model first"
 *   2. the popup downloads the writing model alone, which then loads by
 *      itself; a rewrite is refused until the style it needs is there, which
 *      the popup downloads next
 *   3. one full rewrite on the welcome page's practice box: foreground
 *      controller → service worker → offscreen document → inference worker
 *      (onnxruntime-web, base model plus the style's adapter, read from the
 *      extension's private storage) → the finished rewrite shown with
 *      Replace → Enter puts it in the textarea → Undo restores it. Where the browser offers a GPU this first rewrite runs
 *      there, the default
 *   3b. the live preview: the rewrite shows as it is written, stays in the
 *      same card once finished (where the reader left it) until Replace, the
 *      field changes once; the settings switch turns it off (the text is then
 *      replaced at once); interleaved timed runs with it on and off
 *   4. the popup's settings move the model to the processor (the engine
 *      restarts and reloads it), a style running on the fallback adapter
 *      and a keyboard shortcut rewrite there; a bigger share of the
 *      processor restarts it with more threads; then back to the GPU
 *   5. a press and hold on the practice box's selected text opens the style
 *      menu (Esc and a click elsewhere close it); a style picked there
 *      rewrites the text
 *   6. the idle timer frees the model; the next rewrite says how long waking
 *      it took, and its "keep it awake longer" opens the popup's settings
 *      at that setting
 *   7. settings remove the style, then the writing model
 *
 * Progress and cancelling are covered by the unit tests: from a local server
 * the whole download finishes too quickly to be interrupted reliably.
 *
 * Native context menus cannot be clicked by automation, so the run starts
 * where a menu click hands over: the service worker's "start job" message
 * to the frame. Everything after that is the production path.
 */
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const dist = fileURLToPath(new URL('../../../dist', import.meta.url));
const { url: modelUrl } = JSON.parse(readFileSync(fileURLToPath(new URL('../../../model.source.json', import.meta.url)), 'utf8'));
const ORIGINAL = 'hey can u send me the report by tmrw, its kinda urgent';

function check(condition, message) {
  if (!condition) throw new Error(`E2E check failed: ${message}`);
  console.log(`  ✓ ${message}`);
}

try {
  const response = await fetch(new URL('model.json', modelUrl), { signal: AbortSignal.timeout(5000) });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  await response.body?.cancel();
} catch (error) {
  console.error(`The model is not being served at ${modelUrl} (${error.message}).`);
  console.error('Host the model folder there first (see model.source.json), then run the e2e again.');
  process.exit(1);
}

const profile = mkdtempSync(join(tmpdir(), 'echomebetter-e2e-'));

/** Whether the browser offers extension pages a hardware GPU (what the extension checks too). */
function gpuOffered(page) {
  return page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    return Boolean(adapter) && adapter.info?.isFallbackAdapter !== true && adapter.isFallbackAdapter !== true;
  });
}

/** Entries in the extension's private model folder (OPFS), read from one of its pages. */
function storedModelFiles(page) {
  return page.evaluate(async () => {
    const root = await navigator.storage.getDirectory();
    const folder = await root.getDirectoryHandle('model', { create: true });
    const names = [];
    for await (const name of folder.keys()) names.push(name);
    return names;
  });
}

const context = await chromium.launchPersistentContext(profile, {
  channel: 'chromium',
  headless: true,
  args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`],
});

try {
  const worker = context.serviceWorkers()[0] ?? (await context.waitForEvent('serviceworker'));
  const extensionId = new URL(worker.url()).host;
  console.log(`extension ${extensionId} loaded`);
  const badge = () => worker.evaluate(() => chrome.action.getBadgeText({}));

  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${extensionId}/ui/popup/popup.html`);
  const downloadButton = popup.getByRole('button', { name: /^Download · / });
  await downloadButton.waitFor({ timeout: 10_000 });
  check(true, `popup offers the download on a fresh profile ("${await downloadButton.textContent()}")`);
  const professionalChoice = popup.getByRole('checkbox', { name: /Professional style/ });
  check(await professionalChoice.isChecked(), 'the Professional style is chosen along with the writing model');
  check(await popup.getByText(modelUrl).isVisible(), `popup shows the source URL (${modelUrl})`);
  check((await badge()) === '!', 'toolbar icon is flagged until the model is downloaded');

  const welcome = await context.newPage();
  const consoleErrors = [];
  welcome.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  await welcome.goto(`chrome-extension://${extensionId}/ui/welcome/welcome.html`);
  await welcome.waitForSelector('#playground');
  check(await welcome.evaluate(() => globalThis.crossOriginIsolated === true), 'extension pages are cross-origin isolated (threads available)');
  const gpu = await gpuOffered(welcome);
  const cores = await welcome.evaluate(() => navigator.hardwareConcurrency);
  console.log(`  this browser ${gpu ? 'offers a GPU' : 'offers no GPU'}; ${cores} cores`);
  const speed = welcome.getByRole('region', { name: 'Speed and power' });
  await speed.getByLabel(gpu ? 'GPU power' : 'Processor use').waitFor({ timeout: 10_000 });
  if (gpu) check(await speed.getByRole('radio', { name: /GPU/ }).isChecked(), 'the welcome page offers GPU or CPU, with the GPU chosen');
  else check((await speed.getByRole('radiogroup').count()) === 0, 'without a GPU the welcome page offers no GPU choice, only the processor share');

  await welcome.evaluate((text) => {
    const box = document.querySelector('#playground');
    box.value = text;
    box.focus();
    box.setSelectionRange(0, text.length);
  }, ORIGINAL);

  // The extension has no "tabs" permission, so the page reports its own tab id.
  const tabId = await welcome.evaluate(async () => (await chrome.tabs.getCurrent()).id);
  const startJob = (jobId, style) =>
    worker.evaluate(
      ([id, job, chosen]) => chrome.tabs.sendMessage(id, { kind: 'echo/start-job', jobId: job, style: chosen }, { frameId: 0 }),
      [tabId, jobId, style],
    );
  /** The card holding a finished rewrite of `style`, waiting for Replace or Discard. */
  const readyCard = (style) => welcome.locator('echomebetter-overlay').getByRole('status').filter({ hasText: `Rewrite ready · ${style}` });
  /** Wait for the finished rewrite, then accept it the way a user would: Enter, still in the field. */
  const replaceWithEnter = async (style, timeout = 120_000) => {
    await readyCard(style).getByRole('button', { name: 'Replace' }).waitFor({ timeout });
    await welcome.keyboard.press('Enter');
  };

  console.log('before the download:');
  await startJob('e2e-0', 'professional');
  const refusal = welcome.locator('echomebetter-overlay').getByRole('alert');
  await refusal.waitFor({ timeout: 30_000 });
  check(/Download the writing model first/.test(await refusal.textContent()), 'a rewrite explains that the model must be downloaded first');
  check((await welcome.evaluate(() => document.querySelector('#playground').value)) === ORIGINAL, 'the text was left untouched');

  console.log('download:');
  await professionalChoice.uncheck();
  let downloadStarted = Date.now();
  await downloadButton.click();
  const modelStatus = popup.getByRole('region', { name: 'Writing model status' });
  await modelStatus.waitFor({ timeout: 600_000 });
  console.log(`  writing model: download and verification took ${((Date.now() - downloadStarted) / 1000).toFixed(1)}s`);
  await modelStatus.getByText('Ready', { exact: true }).waitFor({ timeout: 240_000 });
  check((await popup.getByRole('button', { name: 'Load now' }).count()) === 0, 'the downloaded model loaded by itself, with no "Load now" to click');
  check((await storedModelFiles(popup)).length === 4, 'the writing model alone is stored (record and three files)');
  check((await badge()) === '!', 'toolbar icon stays flagged while no style is downloaded');

  await startJob('e2e-style', 'professional');
  await welcome.locator('echomebetter-overlay').getByText(/Download the Professional style first/).waitFor({ timeout: 30_000 });
  check(true, 'a rewrite explains that the style must be downloaded first');

  downloadStarted = Date.now();
  await popup.getByRole('button', { name: /^Download the Professional style/ }).click();
  await popup.getByRole('button', { name: /^Download the Professional style/ }).waitFor({ state: 'detached', timeout: 600_000 });
  console.log(`  Professional style: download and verification took ${((Date.now() - downloadStarted) / 1000).toFixed(1)}s`);
  const stored = await storedModelFiles(popup);
  check(stored.includes('installed.json') && stored.length === 6, `model and style stored in the extension's private storage (${stored.length} entries)`);
  // The refused job above reached the worker, which loaded the base before refusing it: the status may read Ready.
  await popup.getByRole('region', { name: 'Writing model status' }).waitFor({ timeout: 30_000 });
  check((await badge()) === '', 'toolbar flag cleared');
  check((await popup.getByText('Preview').count()) === 4, 'the styles still in training are marked as running on Professional for now');

  console.log('rewrite:');
  await welcome.evaluate(() => {
    const box = document.querySelector('#playground');
    box.focus();
    box.setSelectionRange(0, box.value.length);
  });

  const started = Date.now();
  const reply = await startJob('e2e-1', 'professional');
  check(reply?.ok === true, 'page accepted the job');

  const loader = welcome.locator('echomebetter-overlay').locator('[role="status"]').first();
  await loader.waitFor({ state: 'visible', timeout: 10_000 });
  check(true, `pointer loader visible: "${await loader.getAttribute('aria-label')}"`);
  check(await welcome.evaluate(() => document.documentElement.hasAttribute('data-echomebetter-busy')), 'busy pointer applied to the page');

  await readyCard('Professional').getByRole('button', { name: 'Replace' }).waitFor({ timeout: 240_000 });
  check((await welcome.evaluate(() => document.querySelector('#playground').value)) === ORIGINAL, 'the finished rewrite waits in its card; the text is not replaced yet');
  check(/Took \d/.test(await readyCard('Professional').textContent()), 'the card says how long the rewrite took');
  await welcome.keyboard.press('Enter');
  await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, ORIGINAL, { timeout: 10_000 });
  check(true, 'Enter in the field replaced the text with the rewrite');
  const rewritten = await welcome.evaluate(() => document.querySelector('#playground').value);
  console.log(`  rewrite (${((Date.now() - started) / 1000).toFixed(1)}s incl. model load): ${JSON.stringify(rewritten)}`);
  check(rewritten.trim().length > 0, 'textarea holds the rewrite');
  check(!(await welcome.evaluate(() => document.documentElement.hasAttribute('data-echomebetter-busy'))), 'busy pointer removed');

  const status = await worker.evaluate(() => chrome.storage.session.get('engineStatus'));
  check(status.engineStatus?.state === 'ready', `engine status is ready (${status.engineStatus?.model?.displayName})`);
  check(status.engineStatus.runningOn.processor === (gpu ? 'gpu' : 'cpu'), `the rewrite ran on the ${gpu ? 'GPU' : 'processor'} by default`);
  const firstRewrite = rewritten;

  const undo = welcome.locator('echomebetter-overlay').getByRole('button', { name: 'Undo' });
  await undo.click();
  check((await welcome.evaluate(() => document.querySelector('#playground').value)) === ORIGINAL, 'Undo restored the original text');

  console.log('live preview:');
  const LONG = 'hey team, quick update on the launch. design is mostly done but we still need the final copy from marketing, and qa found a couple of bugs in checkout that we should fix before friday. can someone own the release notes too? thx';
  /**
   * One rewrite of `text`, timed and watched inside the page: from the loader
   * appearing to the rewrite being ready (the card offering Replace, or with
   * the live preview off the field changing; both on the page's clock, so the
   * test driver's own latency is not counted), with every preview text shown,
   * the finished card, and every change the field saw. A finished rewrite
   * waiting in its card is then accepted with Enter.
   */
  const watchedRewrite = async (jobId, text) => {
    await welcome.evaluate((original) => {
      const box = document.querySelector('#playground');
      box.value = original;
      box.focus();
      box.setSelectionRange(0, box.value.length);
      const root = document.querySelector('echomebetter-overlay')?.shadowRoot;
      const seen = { previews: [], inputs: 0, loaderAt: null, readyAt: null, together: false, sameCard: null, review: null };
      let card = null;
      globalThis.__echoWatch = seen;
      const look = () => {
        const shadow = document.querySelector('echomebetter-overlay')?.shadowRoot;
        if (!shadow) return;
        if (seen.loaderAt === null && shadow.querySelector('[role="status"][aria-label]')) seen.loaderAt = performance.now();
        const preview = shadow.querySelector('.echo-scroll[aria-hidden="true"] p');
        const textNow = preview?.textContent ?? null;
        if (textNow && textNow !== seen.previews.at(-1)) seen.previews.push(textNow);
        // The pointer loader carries an aria-label; the card does not.
        if (preview && shadow.querySelector('[role="status"][aria-label]')) seen.together = true;
        if (preview) card ??= preview.closest('[role="status"]');
        const ready = [...shadow.querySelectorAll('[role="status"]')].find((node) => /Rewrite ready/.test(node.textContent));
        if (ready && !seen.review) {
          seen.readyAt ??= performance.now();
          seen.sameCard = ready === card;
          seen.review = { text: ready.querySelector('.echo-scroll p').textContent, card: ready.textContent, field: box.value };
        }
      };
      globalThis.__echoObserver?.disconnect();
      const observer = new MutationObserver(look);
      if (root) observer.observe(root, { subtree: true, childList: true, characterData: true, attributes: true });
      else observer.observe(document.documentElement, { subtree: true, childList: true });
      globalThis.__echoObserver = observer;
      seen.inputs = [];
      box.addEventListener('input', (event) => {
        seen.inputs.push({ at: event.timeStamp, value: box.value });
        seen.readyAt ??= performance.now();
      });
    }, text);
    await startJob(jobId, 'professional');
    await welcome.waitForFunction(() => globalThis.__echoWatch.readyAt !== null, null, { timeout: 120_000, polling: 'raf' });
    if (await welcome.evaluate(() => globalThis.__echoWatch.review !== null)) await welcome.keyboard.press('Enter');
    await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, text, { timeout: 10_000, polling: 'raf' });
    await welcome.waitForTimeout(150);
    return welcome.evaluate(() => {
      globalThis.__echoObserver?.disconnect();
      return { ...globalThis.__echoWatch, value: document.querySelector('#playground').value };
    });
  };
  const setLivePreview = (on) =>
    worker.evaluate(async (value) => {
      const { settings } = await chrome.storage.local.get('settings');
      await chrome.storage.local.set({ settings: { ...settings, livePreview: value } });
    }, on);

  // On by default: the rewrite shows as it is written, and the field changes once, to the finished text.
  const streamed = await watchedRewrite('e2e-live-1', LONG);
  console.log(`  ${streamed.previews.length} preview updates, the first ${JSON.stringify(streamed.previews[0])}`);
  check(streamed.previews.length >= 2, `the rewrite showed as it was written (${streamed.previews.length} updates)`);
  check(
    streamed.previews.every((text, index) => index === 0 || text.length >= streamed.previews[index - 1].length),
    'each preview carries on from the one before',
  );
  /** The field was written once, in one go (one edit can fire an input event per line), and only with the finished text. */
  const writtenOnce = ({ inputs, value }) => inputs.length > 0 && inputs.every((input) => input.value === value) && inputs.at(-1).at - inputs[0].at < 50;
  check(writtenOnce(streamed), `the field changed once, to the finished text (${streamed.inputs.length} input events, all within one edit)`);
  check(streamed.value !== LONG && streamed.value.trim().length > 0, 'the field holds the finished rewrite');
  check(streamed.previews.every((text) => text.trim().length > 0 && !text.includes('\uFFFD')), 'no empty or broken preview was shown');
  check(!streamed.together, 'the loader gave way to the card at the first words: never both on the page');
  check(streamed.sameCard === true, 'the card that showed the rewrite kept it, whole, once it was finished');
  check(streamed.review.text.startsWith(streamed.previews.at(-1)) && streamed.review.text === streamed.value, 'the finished text carries on from the last words shown, and is what went into the field');
  check(streamed.review.field === LONG, 'nothing was replaced until Replace');
  check(/Took \d/.test(streamed.review.card), 'the finished card says how long the rewrite took');
  const resultCard = welcome.locator('echomebetter-overlay').getByRole('status').filter({ hasText: 'Replaced · Professional' });
  check((await resultCard.getByRole('button', { name: 'Undo' }).count()) === 1, 'after Replace the card offers Undo');

  // The switch in settings turns it off and on.
  await popup.getByRole('button', { name: 'Settings' }).click();
  const liveSwitch = popup.getByRole('switch', { name: 'Live preview' });
  await liveSwitch.waitFor({ timeout: 10_000 });
  check((await liveSwitch.getAttribute('aria-checked')) === 'true', 'settings show the live preview on by default');
  await liveSwitch.click();
  await popup.waitForTimeout(200);
  check((await worker.evaluate(() => chrome.storage.local.get('settings'))).settings.livePreview === false, 'the switch turned the live preview off');
  const withoutPreview = await watchedRewrite('e2e-live-2', LONG);
  check(
    withoutPreview.previews.length === 0 && withoutPreview.review === null && writtenOnce(withoutPreview) && withoutPreview.inputs.length === streamed.inputs.length,
    'with it off, the rewrite goes in at once, as before',
  );
  await liveSwitch.click();
  await popup.waitForTimeout(200);
  check((await worker.evaluate(() => chrome.storage.local.get('settings'))).settings.livePreview === true, 'and back on');
  await popup.getByRole('button', { name: 'Back' }).click();

  // A paragraph: the card widens to fit it (up to 70% of the window), then scrolls, and the reader can scroll back.
  const PARAGRAPH = Array.from({ length: 4 }, () => 'hey team, quick update on the launch. design is mostly done but we still need the final copy from marketing, and qa found a couple of bugs in checkout that we should fix before friday.').join(' ');
  await welcome.evaluate((original) => {
    const box = document.querySelector('#playground');
    box.value = original;
    box.focus();
    box.setSelectionRange(0, box.value.length);
  }, PARAGRAPH);
  const viewportWidth = await welcome.evaluate(() => window.innerWidth);
  const sampleCard = () =>
    welcome.evaluate(() => {
      const shadow = document.querySelector('echomebetter-overlay')?.shadowRoot;
      const text = shadow?.querySelector('.echo-scroll[aria-hidden="true"] p');
      if (!text) return null;
      const card = text.closest('[role="status"]');
      const scroller = text.parentElement;
      const box = card.getBoundingClientRect();
      return {
        width: box.width,
        left: box.left,
        right: box.right,
        scrollable: scroller.scrollHeight > scroller.clientHeight + 1,
        height: scroller.clientHeight,
        atEnd: scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= 4,
        top: scroller.scrollTop,
        pageScroll: window.scrollY,
      };
    });
  await startJob('e2e-wide', 'professional');
  const samples = [];
  let reader = null;
  const wideDeadline = Date.now() + 120_000;
  while (Date.now() < wideDeadline) {
    const now = await sampleCard();
    if (now) {
      samples.push({ ...now, beforeReading: reader === null });
      // Once it scrolls: wheel up over the card, as a reader would, and see it stay put while text keeps coming.
      if (now.scrollable && reader === null) {
        const card = welcome.locator('echomebetter-overlay').locator('[aria-hidden="true"] > p').first();
        await card.hover();
        const before = await sampleCard();
        await welcome.mouse.wheel(0, -400);
        await welcome.waitForTimeout(120);
        const after = await sampleCard();
        await welcome.waitForTimeout(600);
        const later = await sampleCard();
        reader = { before, after, later };
      }
    } else if (samples.length > 0) {
      break;
    }
    await welcome.waitForTimeout(100);
  }
  await readyCard('Professional')
    .getByRole('button', { name: 'Replace' })
    .waitFor({ timeout: 60_000 })
    .catch(async (error) => {
      const shown = await welcome.evaluate(() => document.querySelector('echomebetter-overlay')?.shadowRoot?.textContent ?? '(no overlay)');
      throw new Error(`the paragraph was not rewritten; the page showed: ${shown.slice(0, 300)} (${error.message.split('\n')[0]})`);
    });
  // Finished while the reader was reading further up: the text stays where they were.
  const finished = await welcome.evaluate(() => {
    const scroller = document.querySelector('echomebetter-overlay').shadowRoot.querySelector('.echo-scroll');
    return { top: scroller.scrollTop, atEnd: scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= 4 };
  });
  await welcome.keyboard.press('Enter');
  await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, PARAGRAPH, { timeout: 10_000 });
  const widths = samples.map((sample) => sample.width);
  console.log(`  card widths while writing: ${Math.round(Math.min(...widths))} → ${Math.round(Math.max(...widths))} px of ${viewportWidth}`);
  check(Math.min(...widths) >= 340 && Math.max(...widths) > Math.min(...widths) + 50, 'the card widened as the paragraph grew');
  check(Math.max(...widths) <= viewportWidth * 0.7 + 1, 'and stopped at 70% of the window');
  check(samples.every((sample) => sample.left >= 0 && sample.right <= viewportWidth), 'always fully inside the window');
  check(samples.some((sample) => sample.scrollable), 'a long rewrite scrolls inside the card');
  check(Math.max(...samples.map((sample) => sample.height)) <= 120, 'the text area stays within its height limit');
  check(samples.filter((sample) => sample.scrollable && sample.beforeReading).every((sample) => sample.atEnd), 'the newest text stays in view while it arrives');
  check(reader !== null && reader.after.top < reader.before.top && !reader.after.atEnd, 'scrolling up over the card moved the text');
  check(reader.later.top <= reader.after.top + 1 && !reader.later.atEnd, 'new text did not pull the reader back down');
  check(reader.after.pageScroll === reader.before.pageScroll, 'the page behind did not scroll');
  check(Math.abs(finished.top - reader.later.top) <= 1 && !finished.atEnd, 'when it finished, the text stayed where the reader was');

  // Does showing it slow the rewrite down? Interleaved runs, on the page's clock.
  const RUNS = Number(process.env.E2E_TIMING_RUNS ?? 6);
  const times = { on: [], off: [] };
  for (let run = 0; run < RUNS; run++) {
    for (const on of run % 2 ? [true, false] : [false, true]) {
      await setLivePreview(on);
      const result = await watchedRewrite(`e2e-time-${run}-${on}`, LONG);
      check(result.loaderAt !== null && result.readyAt !== null, 'the run was timed');
      times[on ? 'on' : 'off'].push(result.readyAt - result.loaderAt);
    }
  }
  await setLivePreview(true);
  const median = (values) => [...values].sort((x, y) => x - y)[Math.floor(values.length / 2)];
  const fmt = (values) => values.map((value) => Math.round(value)).join(', ');
  console.log(`  preview off: ${fmt(times.off)} ms (median ${Math.round(median(times.off))})`);
  console.log(`  preview on:  ${fmt(times.on)} ms (median ${Math.round(median(times.on))})`);
  // Paired by run (each pair ran back to back), so a machine slowing down over the runs cancels out.
  const cost = median(times.on.map((on, run) => on - times.off[run]));
  const share = cost / median(times.off);
  console.log(`  difference, paired: ${cost >= 0 ? '+' : ''}${Math.round(cost)} ms (${(share * 100).toFixed(1)}%)`);
  // Reported, and failed only when gross: on a shared machine one run varies by a few hundred milliseconds.
  // The bound that matters (a few updates a second, not one per token) is held by the engine host's unit test.
  check(share <= 0.15, 'the live preview does not slow the rewrite down noticeably');

  /** Engine status once it is ready on `processor` again after a settings change restarted it. */
  const readyOn = async (processor) => {
    const deadline = Date.now() + 240_000;
    for (;;) {
      const { engineStatus } = await worker.evaluate(() => chrome.storage.session.get('engineStatus'));
      if (engineStatus?.state === 'ready' && engineStatus.runningOn.processor === processor) return engineStatus;
      if (engineStatus?.state === 'error') throw new Error(`E2E check failed: the engine failed to load: ${engineStatus.message}`);
      if (Date.now() > deadline) throw new Error(`E2E check failed: the engine never became ready on the ${processor}`);
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  };

  console.log('performance settings:');
  await popup.getByRole('button', { name: 'Settings' }).click();
  const performance = popup.getByRole('region', { name: 'Performance' });
  await performance.getByLabel(gpu ? 'GPU power' : 'Processor use').waitFor({ timeout: 10_000 });
  if (gpu) {
    let restarted = Date.now();
    await performance.getByText('CPU', { exact: true }).click();
    const onCpu = await readyOn('cpu');
    console.log(`  restart onto the processor and reload took ${((Date.now() - restarted) / 1000).toFixed(1)}s`);
    check(onCpu.runningOn.threads >= 1, `choosing CPU restarted the loaded model on the processor (${onCpu.runningOn.threads} threads)`);
    await performance.getByText(`Running on the processor now, with ${onCpu.runningOn.threads} threads.`).waitFor({ timeout: 10_000 });
    check(true, 'the settings say where the model runs now');

    await welcome.evaluate((original) => {
      const box = document.querySelector('#playground');
      box.value = original;
      box.focus();
      box.setSelectionRange(0, box.value.length);
    }, ORIGINAL);
    const cpuStarted = Date.now();
    await startJob('e2e-cpu', 'professional');
    await replaceWithEnter('Professional', 240_000);
    await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, ORIGINAL, { timeout: 240_000 });
    const onProcessor = await welcome.evaluate(() => document.querySelector('#playground').value);
    console.log(`  warm rewrite on the processor took ${((Date.now() - cpuStarted) / 1000).toFixed(1)}s: ${JSON.stringify(onProcessor)}`);
    console.log(`  the GPU and the processor wrote ${onProcessor === firstRewrite ? 'the same rewrite' : 'different rewrites'}`);
    check(onProcessor.trim().length > 0, 'the processor rewrote the text');
    await welcome.locator('echomebetter-overlay').getByRole('button', { name: 'Undo' }).click();
  }
  const usage = performance.getByLabel('Processor use');
  const balancedThreads = (await worker.evaluate(() => chrome.storage.session.get('engineStatus'))).engineStatus.runningOn.threads;
  await usage.selectOption('maximum');
  const maximum = await (async () => {
    const deadline = Date.now() + 240_000;
    for (;;) {
      const { engineStatus } = await worker.evaluate(() => chrome.storage.session.get('engineStatus'));
      if (engineStatus?.state === 'ready' && engineStatus.runningOn.threads !== balancedThreads) return engineStatus;
      if (cores < 4) return engineStatus;
      if (Date.now() > deadline) throw new Error(`E2E check failed: the model was not restarted with more threads (status ${JSON.stringify(engineStatus)})`);
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  })();
  check(
    cores < 4 || maximum.runningOn.threads > balancedThreads,
    `a bigger share of the processor restarted the model with more threads (${balancedThreads} → ${maximum.runningOn.threads})`,
  );
  await usage.selectOption('balanced');
  await popup.getByRole('button', { name: 'Back' }).click();

  // A second rewrite runs on the already-loaded model.
  await welcome.evaluate(() => {
    const box = document.querySelector('#playground');
    box.focus();
    box.setSelectionRange(0, box.value.length);
  });
  const second = Date.now();
  await startJob('e2e-2', 'grammar');
  await replaceWithEnter('Grammar');
  await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, ORIGINAL, { timeout: 120_000 });
  const fallback = await welcome.evaluate(() => document.querySelector('#playground').value);
  console.log(`  warm rewrite with the fallback adapter took ${((Date.now() - second) / 1000).toFixed(1)}s: ${JSON.stringify(fallback)}`);

  // The same rewrite from the keyboard: the welcome page listens for shortcuts itself.
  await welcome.evaluate((original) => {
    const box = document.querySelector('#playground');
    box.value = original;
    box.focus();
    box.setSelectionRange(0, box.value.length);
  }, ORIGINAL);
  const mac = await welcome.evaluate(() => /mac/i.test(navigator.userAgentData?.platform || navigator.platform));
  await welcome.keyboard.press(`${mac ? 'Control' : 'Alt'}+Shift+KeyF`);
  await replaceWithEnter('Friendly');
  check(true, 'the shortcut named its style in the card');
  await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, ORIGINAL, { timeout: 10_000 });
  check(true, `${mac ? '⌃⇧F' : 'Alt+Shift+F'} rewrote the selected text`);

  console.log('press and hold:');
  const overlay = welcome.locator('echomebetter-overlay');
  const styleMenu = overlay.getByRole('menu', { name: 'Rewrite as' });
  /** Select the whole practice text, then press and hold on its first line; resolves once the menu shows, still holding. */
  const holdOnSelection = async () => {
    await welcome.evaluate((original) => {
      const box = document.querySelector('#playground');
      box.value = original;
      box.focus();
      box.setSelectionRange(0, box.value.length);
    }, ORIGINAL);
    const box = await welcome.locator('#playground').boundingBox();
    await welcome.mouse.move(box.x + 40, box.y + 26);
    await welcome.mouse.down();
    await styleMenu.waitFor({ timeout: 5_000 });
  };
  const selectionOfBox = () => welcome.evaluate(() => {
    const box = document.querySelector('#playground');
    return [box.selectionStart, box.selectionEnd, box.value.length];
  });

  /** The menu's own size, and its first icon box and row: what a page's root font size must not change. */
  const menuGeometry = () =>
    welcome.evaluate(() => {
      const shadow = document.querySelector('echomebetter-overlay').shadowRoot;
      const box = (node) => {
        const { width, height } = node.getBoundingClientRect();
        return [Math.round(width * 10) / 10, Math.round(height * 10) / 10];
      };
      const menu = shadow.querySelector('[role="menu"]');
      const row = menu.querySelector('[role="menuitem"]');
      return { menu: box(menu), row: box(row), icon: box(row.firstElementChild) };
    });

  const held = Date.now();
  await holdOnSelection();
  const normalGeometry = await menuGeometry();
  const heldFor = Date.now() - held;
  check(heldFor >= 900, `holding still on selected text opened the style menu (after ${(heldFor / 1000).toFixed(1)}s)`);
  check((await styleMenu.getByRole('menuitem').count()) === 5, 'the menu offers every style');
  await welcome.mouse.up();
  const [selStart, selEnd, length] = await selectionOfBox();
  check(selStart === 0 && selEnd === length, 'letting go kept the text selected');
  await welcome.keyboard.press('Escape');
  await styleMenu.waitFor({ state: 'detached', timeout: 5_000 });
  check((await welcome.evaluate(() => document.querySelector('#playground').value)) === ORIGINAL, 'Esc closed the menu and left the text alone');

  await holdOnSelection();
  await welcome.mouse.up();
  await welcome.mouse.click(10, 10);
  await styleMenu.waitFor({ state: 'detached', timeout: 5_000 });
  check(true, 'a click elsewhere closed the menu');

  // A site that sets its root font size small (html { font-size: 62.5% } is common): the overlay must not shrink with it.
  await welcome.evaluate(() => (document.documentElement.style.fontSize = '10px'));
  await holdOnSelection();
  const smallRootGeometry = await menuGeometry();
  await welcome.mouse.up();
  await welcome.keyboard.press('Escape');
  await styleMenu.waitFor({ state: 'detached', timeout: 5_000 });
  await welcome.evaluate(() => document.documentElement.style.removeProperty('font-size'));
  console.log(`  menu at a 16px root: ${JSON.stringify(normalGeometry)}\n  menu at a 10px root: ${JSON.stringify(smallRootGeometry)}`);
  check(JSON.stringify(smallRootGeometry) === JSON.stringify(normalGeometry), "the menu looks the same on a page whose root font size is 10px");

  await holdOnSelection();
  await welcome.mouse.up();
  await styleMenu.getByRole('menuitem', { name: /Concise/ }).click();
  await replaceWithEnter('Concise');
  await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, ORIGINAL, { timeout: 10_000 });
  await overlay.getByText('Replaced · Concise').waitFor({ timeout: 10_000 });
  check(true, `picking Concise in the menu rewrote the text: ${JSON.stringify(await welcome.evaluate(() => document.querySelector('#playground').value))}`);
  await overlay.getByRole('button', { name: 'Undo' }).click();

  // A plain click on the selection is still a click: no menu.
  await welcome.evaluate(() => {
    const box = document.querySelector('#playground');
    box.focus();
    box.setSelectionRange(0, box.value.length);
  });
  const plain = await welcome.locator('#playground').boundingBox();
  await welcome.mouse.click(plain.x + 40, plain.y + 26);
  await welcome.waitForTimeout(2_500);
  check((await styleMenu.count()) === 0, 'an ordinary click on selected text opens nothing');

  if (gpu) {
    await popup.getByRole('button', { name: 'Settings' }).click();
    const restarted = Date.now();
    await popup.getByRole('region', { name: 'Performance' }).getByText('GPU', { exact: true }).click();
    await readyOn('gpu');
    console.log(`  restart back onto the GPU and reload took ${((Date.now() - restarted) / 1000).toFixed(1)}s`);
    check(true, 'choosing GPU again restarted the model on the GPU');
    await popup.getByRole('button', { name: 'Back' }).click();
  }

  // The finished rewrite sits next to the selected text, not at the bottom of a taller text box.
  await welcome.evaluate(() => {
    const box = document.querySelector('#playground');
    box.style.height = '420px';
    box.value = `${'pls fix this sentense'}\n${'and also thsi one'}${'\n'.repeat(12)}last line`;
    box.scrollTop = 0;
    box.focus();
    box.setSelectionRange(0, box.value.indexOf('\n\n'));
  });
  await startJob('e2e-place', 'grammar');
  const placedCard = welcome.locator('echomebetter-overlay').getByRole('status').filter({ hasText: /Rewrite ready|Looks good/ });
  await placedCard.waitFor({ timeout: 120_000 });
  const gap = await welcome.evaluate(() => {
    const box = document.querySelector('#playground').getBoundingClientRect();
    const card = [...document.querySelector('echomebetter-overlay').shadowRoot.querySelectorAll('[role="status"]')].find((node) => /Rewrite ready|Looks good/.test(node.textContent));
    return { fromTop: card.getBoundingClientRect().top - box.top, boxHeight: box.height };
  });
  check(gap.fromTop < gap.boxHeight / 2, `the card sits beside the two selected lines (${Math.round(gap.fromTop)}px into a ${Math.round(gap.boxHeight)}px text box)`);
  if (/Rewrite ready/.test(await placedCard.textContent())) {
    const before = await welcome.evaluate(() => document.querySelector('#playground').value);
    await welcome.keyboard.press('Escape');
    await placedCard.waitFor({ state: 'detached', timeout: 5_000 });
    check((await welcome.evaluate(() => document.querySelector('#playground').value)) === before, 'Esc discarded the finished rewrite and left the text as it was');
  }

  console.log('waking from an idle rest:');
  // What the idle check finds after an hour without use: the alarm, due now, frees the model. A check
  // that first applies a settings change still owed from above restarts the model instead; then the next one frees it.
  const resting = async () => {
    const { engineStatus, modelResting } = await worker.evaluate(() => chrome.storage.session.get(['engineStatus', 'modelResting']));
    return engineStatus?.state === 'unloaded' && modelResting === true;
  };
  for (let attempt = 0; attempt < 5 && !(await resting()); attempt += 1) {
    await worker.evaluate(async () => {
      await chrome.storage.session.set({ lastActivityAt: 0 });
      await chrome.alarms.create('echomebetter/idle-check', { when: Date.now() + 100 });
    });
    await welcome.waitForTimeout(3_000);
  }
  check(await resting(), 'the idle timer freed the model');
  await welcome.evaluate((original) => {
    const box = document.querySelector('#playground');
    box.style.height = '';
    box.value = original;
    box.focus();
    box.setSelectionRange(0, box.value.length);
  }, ORIGINAL);
  await startJob('e2e-wake', 'professional');
  const wakeNote = overlay.getByText(/^Waking up took .+\. EchoMeBetter rests after 1 hour without use, to keep your computer fast\.$/);
  await wakeNote.waitFor({ timeout: 240_000 });
  check(true, `the rewrite says how long waking up took: "${await wakeNote.textContent()}"`);
  // The toolbar popup opens over the browser window; automation can't drive it as a page, but Chrome lists it.
  const cdp = await context.newCDPSession(welcome);
  const popupTargets = async () =>
    (await cdp.send('Target.getTargets')).targetInfos.filter((target) => target.url.endsWith('/ui/popup/popup.html')).length;
  const popupsBefore = await popupTargets();
  await overlay.getByRole('button', { name: 'Keep it awake longer' }).click();
  for (const deadline = Date.now() + 10_000; (await popupTargets()) === popupsBefore && Date.now() < deadline; ) await welcome.waitForTimeout(100);
  const intentLeft = 'popupIntent' in (await worker.evaluate(() => chrome.storage.session.get('popupIntent')));
  check((await popupTargets()) === popupsBefore + 1 && !intentLeft, '"Keep it awake longer" opened the toolbar popup, which took the request to start at that setting');

  // What that popup shows, on a page automation can read: the same request, taken the same way.
  await worker.evaluate(() => chrome.storage.session.set({ popupIntent: { page: 'settings', focus: 'keep-awake', at: Date.now() } }));
  const settingsPage = await context.newPage();
  await settingsPage.goto(`chrome-extension://${extensionId}/ui/popup/popup.html`);
  const keepAwake = settingsPage.getByLabel(/Free memory when idle for/);
  await keepAwake.waitFor({ timeout: 10_000 });
  check(await keepAwake.evaluate((select) => select === document.activeElement), 'the popup opens on its settings with the keep-awake time focused');
  check(await settingsPage.getByText(/Pick a longer time to keep it awake between rewrites/).isVisible(), 'the settings say why it is worth a longer time');
  await settingsPage.close();

  check(consoleErrors.length === 0, `no console errors on the page${consoleErrors.length ? `: ${consoleErrors.join(' | ')}` : ''}`);

  console.log('remove:');
  await popup.getByRole('button', { name: 'Settings' }).click();
  await popup.getByRole('button', { name: 'Remove professional style' }).click();
  await popup.getByRole('button', { name: 'Remove', exact: true }).click();
  await popup.getByRole('button', { name: 'Remove professional style' }).waitFor({ state: 'detached', timeout: 30_000 });
  check((await storedModelFiles(popup)).length === 4, 'removing the style deleted only its files');
  check((await badge()) === '!', 'toolbar icon is flagged again without a style');
  await popup.getByRole('button', { name: 'Remove writing model' }).click();
  await popup.getByRole('button', { name: 'Remove', exact: true }).click();
  await popup.getByText('Nothing downloaded yet.').waitFor({ timeout: 30_000 });
  check((await storedModelFiles(popup)).length === 0, 'every model file was deleted');
  await popup.getByRole('button', { name: 'Back' }).click();
  await popup.getByText('Model removed. Download it again whenever you want to rewrite text.').waitFor({ timeout: 10_000 });
  check(await downloadButton.isVisible(), 'popup offers the download again');

  await strictCspPageScenario(context);
  console.log('E2E passed');
} finally {
  await context.close();
  rmSync(profile, { recursive: true, force: true });
}

/**
 * The injected foreground bundle on an ordinary website whose CSP forbids
 * inline styles and scripts. Automation cannot click a native context menu
 * (which is what grants activeTab for the real injection), so dist/foreground.js
 * is evaluated directly with a stand-in `chrome.runtime` whose job port
 * answers like the service worker does. What this proves: the bundle runs in
 * a real page, its overlay is styled despite the CSP, and a contenteditable
 * editor gets the rewrite through the native editing path.
 */
async function strictCspPageScenario(context) {
  console.log('strict-CSP page:');
  const page = await context.newPage();
  await page.route('https://echomebetter.test/**', (route) =>
    route.fulfill({
      contentType: 'text/html',
      headers: { 'content-security-policy': "default-src 'none'; style-src 'self'; script-src 'self'" },
      body: '<!doctype html><html><body><div id="editor" contenteditable="true">please   fix this sentense for me thanks</div></body></html>',
    }),
  );
  await page.goto('https://echomebetter.test/compose');
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));

  await page.evaluate(() => {
    globalThis.chrome = {
      runtime: {
        id: 'e2e',
        connect() {
          const listeners = [];
          return {
            postMessage(message) {
              if (message.kind !== 'job/request') return;
              setTimeout(() => listeners.forEach((l) => l({ kind: 'job/phase', jobId: message.jobId, phase: 'rewriting' })), 50);
              // The second job is watched as it is written (live preview on); the first is not.
              if (message.jobId === 'csp-2') setTimeout(() => listeners.forEach((l) => l({ kind: 'job/partial', jobId: message.jobId, text: 'Please fix' })), 200);
              setTimeout(() => listeners.forEach((l) => l({ kind: 'job/done', jobId: message.jobId, text: 'Please fix this sentence for me. Thanks!' })), 400);
            },
            disconnect() {},
            onMessage: { addListener: (listener) => listeners.push(listener) },
            onDisconnect: { addListener() {} },
          };
        },
        onMessage: { addListener: (listener) => (globalThis.__echoDeliver = listener) },
      },
    };
  });
  await page.evaluate(readFileSync(join(dist, 'foreground.js'), 'utf8'));

  await page.evaluate(() => {
    const editor = document.getElementById('editor');
    editor.focus();
    const range = document.createRange();
    range.selectNodeContents(editor);
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  });
  const reply = await page.evaluate(
    () => new Promise((resolve) => globalThis.__echoDeliver({ kind: 'echo/start-job', jobId: 'csp-1', style: 'grammar' }, {}, resolve)),
  );
  check(reply.ok === true, 'foreground accepted the job on a strict-CSP page');

  const loaderBackground = await page.evaluate(() => {
    const loader = document.querySelector('echomebetter-overlay').shadowRoot.querySelector('[role="status"]');
    return loader ? getComputedStyle(loader).backgroundColor : null;
  });
  check(loaderBackground === 'rgb(23, 18, 43)', `overlay is styled despite style-src (loader background ${loaderBackground})`);

  await page.waitForFunction(() => document.getElementById('editor').textContent.startsWith('Please fix'), null, { timeout: 5000 });
  check(true, 'contenteditable received the rewrite');
  await page.evaluate(() => document.execCommand('undo'));
  check(
    (await page.evaluate(() => document.getElementById('editor').textContent)) === 'please   fix this sentense for me thanks',
    'native undo (Ctrl+Z) restores the original in the editor',
  );

  // Watched as it was written: the finished rewrite waits; Enter in the editor replaces the text, and adds no line.
  await page.evaluate(() => {
    const editor = document.getElementById('editor');
    editor.focus();
    const range = document.createRange();
    range.selectNodeContents(editor);
    getSelection().removeAllRanges();
    getSelection().addRange(range);
  });
  await page.evaluate(() => new Promise((resolve) => globalThis.__echoDeliver({ kind: 'echo/start-job', jobId: 'csp-2', style: 'grammar' }, {}, resolve)));
  await page.locator('echomebetter-overlay').getByRole('button', { name: 'Replace' }).waitFor({ timeout: 5000 });
  check((await page.evaluate(() => document.getElementById('editor').textContent)) === 'please   fix this sentense for me thanks', 'the editor keeps its text while the rewrite waits');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => document.getElementById('editor').textContent.startsWith('Please fix'), null, { timeout: 5000 });
  check(
    await page.evaluate(() => {
      const editor = document.getElementById('editor');
      return editor.textContent === 'Please fix this sentence for me. Thanks!' && editor.querySelectorAll('br, div, p').length === 0;
    }),
    'Enter replaced the text in the contenteditable editor without adding a line break',
  );
  check(errors.length === 0, `no page errors${errors.length ? `: ${errors.join(' | ')}` : ''}`);
  await page.close();
}
