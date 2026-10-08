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
 *   2. the popup downloads the writing model alone; a rewrite is then refused
 *      until the style it needs is there, which the popup downloads next
 *   3. one full rewrite on the welcome page's practice box: foreground
 *      controller → service worker → offscreen document → inference worker
 *      (onnxruntime-web, base model plus the style's adapter, read from the
 *      extension's private storage) → text replaced in the textarea → Undo
 *      restores it. Where the browser offers a GPU this first rewrite runs
 *      there, the default
 *   4. the popup's settings move the model to the processor (the engine
 *      restarts and reloads it), a style running on the fallback adapter
 *      and a keyboard shortcut rewrite there; a bigger share of the
 *      processor restarts it with more threads; then back to the GPU
 *   5. settings remove the style, then the writing model
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
  await popup.getByRole('button', { name: 'Load now' }).waitFor({ timeout: 600_000 });
  console.log(`  writing model: download and verification took ${((Date.now() - downloadStarted) / 1000).toFixed(1)}s`);
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

  await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, ORIGINAL, {
    timeout: 240_000,
    polling: 250,
  });
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
  await welcome.waitForFunction((original) => document.querySelector('#playground').value !== original, ORIGINAL, { timeout: 120_000 });
  check(true, `${mac ? '⌃⇧F' : 'Alt+Shift+F'} rewrote the selected text`);
  const keyboardToast = welcome.locator('echomebetter-overlay').getByText('Rewritten · Friendly');
  await keyboardToast.waitFor({ timeout: 10_000 });
  check(true, 'the shortcut named its style in the toast');

  if (gpu) {
    await popup.getByRole('button', { name: 'Settings' }).click();
    const restarted = Date.now();
    await popup.getByRole('region', { name: 'Performance' }).getByText('GPU', { exact: true }).click();
    await readyOn('gpu');
    console.log(`  restart back onto the GPU and reload took ${((Date.now() - restarted) / 1000).toFixed(1)}s`);
    check(true, 'choosing GPU again restarted the model on the GPU');
    await popup.getByRole('button', { name: 'Back' }).click();
  }

  // The result toast sits next to the selected text, not at the bottom of a taller text box.
  await welcome.evaluate(() => {
    const box = document.querySelector('#playground');
    box.style.height = '420px';
    box.value = `${'pls fix this sentense'}\n${'and also thsi one'}${'\n'.repeat(12)}last line`;
    box.scrollTop = 0;
    box.focus();
    box.setSelectionRange(0, box.value.indexOf('\n\n'));
  });
  await startJob('e2e-place', 'grammar');
  const placedToast = welcome.locator('echomebetter-overlay').getByRole('status').filter({ hasText: /Rewritten|Looks good/ });
  await placedToast.waitFor({ timeout: 120_000 });
  const gap = await welcome.evaluate(() => {
    const box = document.querySelector('#playground').getBoundingClientRect();
    const toast = [...document.querySelector('echomebetter-overlay').shadowRoot.querySelectorAll('[role="status"]')].find((node) => /Rewritten|Looks good/.test(node.textContent));
    return { fromTop: toast.getBoundingClientRect().top - box.top, boxHeight: box.height };
  });
  check(gap.fromTop < gap.boxHeight / 2, `toast sits beside the two selected lines (${Math.round(gap.fromTop)}px into a ${Math.round(gap.boxHeight)}px text box)`);

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
  check(errors.length === 0, `no page errors${errors.length ? `: ${errors.join(' | ')}` : ''}`);
  await page.close();
}
