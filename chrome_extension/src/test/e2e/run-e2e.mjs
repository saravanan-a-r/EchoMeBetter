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
 *      restores it; then a style running on the fallback adapter, and the
 *      same through a keyboard shortcut
 *   4. settings remove the style, then the writing model
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

  const undo = welcome.locator('echomebetter-overlay').getByRole('button', { name: 'Undo' });
  await undo.click();
  check((await welcome.evaluate(() => document.querySelector('#playground').value)) === ORIGINAL, 'Undo restored the original text');

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
