import { describe, expect, jest, test } from '@jest/globals';
import { styleMenuId } from '../../../background/contextMenus';
import { launchRewrite, launchStyle, launchStyleMenu, type LauncherDeps } from '../../../background/rewriteLauncher';
import type { ErrorPayload } from '../../../shared/errors';

function deps(options: { injectFails?: (frameId: number) => boolean; problem?: ErrorPayload } = {}) {
  const calls: string[] = [];
  const value: LauncherDeps = {
    scripting: {
      executeScript: jest.fn(async ({ target }: { target: { frameIds?: number[] } }) => {
        const frameId = target.frameIds?.[0] ?? 0;
        calls.push(`inject:${frameId}`);
        if (options.injectFails?.(frameId)) throw new Error('Cannot access contents of the page');
        return [];
      }),
    } as unknown as LauncherDeps['scripting'],
    tabs: {
      sendMessage: jest.fn(async (_tabId: number, message: { kind: string }, { frameId }: { frameId: number }) => {
        calls.push(`${message.kind}:${frameId}`);
        return { ok: true };
      }),
    },
    action: {
      setBadgeText: jest.fn(async ({ text }: { text: string }) => void calls.push(`badge:${text}`)),
      setBadgeBackgroundColor: jest.fn(async () => undefined),
      setTitle: jest.fn(async () => undefined),
    } as unknown as LauncherDeps['action'],
    extensionOrigin: 'chrome-extension://abc/',
    prepareEngine: jest.fn(() => void calls.push('prepare')),
    checkStyle: async () => options.problem ?? null,
    newJobId: () => 'job-1',
  };
  return { value, calls };
}

const tab = { id: 7, url: 'https://mail.example.com/compose' } as chrome.tabs.Tab;

describe('launchRewrite', () => {
  test('injects into the clicked frame, warms the engine, then starts the job there', async () => {
    const { value, calls } = deps();
    await launchRewrite({ menuItemId: styleMenuId('professional'), frameId: 3 }, tab, value);
    expect(calls).toEqual(['inject:3', 'prepare', 'badge:', 'echo/start-job:3']);
    expect(value.tabs.sendMessage).toHaveBeenCalledWith(7, { kind: 'echo/start-job', jobId: 'job-1', style: 'professional' }, { frameId: 3 });
  });

  test("the extension's own pages are not injected into (they load the foreground themselves)", async () => {
    const { value, calls } = deps();
    await launchRewrite({ menuItemId: styleMenuId('concise') }, { id: 7, url: 'chrome-extension://abc/ui/welcome/welcome.html' } as chrome.tabs.Tab, value);
    expect(calls).toEqual(['prepare', 'badge:', 'echo/start-job:0']);
  });

  test('an unreachable cross-origin frame is explained in the top frame, and no model is loaded', async () => {
    const { value, calls } = deps({ injectFails: (frameId) => frameId !== 0 });
    await launchRewrite({ menuItemId: styleMenuId('grammar'), frameId: 4 }, tab, value);
    expect(calls).toEqual(['inject:4', 'inject:0', 'echo/notice:0']);
  });

  test('a page that cannot be scripted at all gets a toolbar badge', async () => {
    const { value, calls } = deps({ injectFails: () => true });
    await launchRewrite({ menuItemId: styleMenuId('grammar'), frameId: 0 }, tab, value);
    expect(calls).toEqual(['inject:0', 'badge:!']);
  });

  test.each<ErrorPayload>([{ code: 'MODEL_NOT_DOWNLOADED' }, { code: 'STYLE_NOT_DOWNLOADED', details: { style: 'Friendly', adapter: 'Professional' } }])(
    'a style that cannot run yet ($code) is explained in the clicked frame, and nothing loads',
    async (problem) => {
      const { value, calls } = deps({ problem });
      await launchRewrite({ menuItemId: styleMenuId('friendly'), frameId: 2 }, tab, value);
      expect(calls).toEqual(['inject:2', 'echo/notice:2']);
      expect(value.tabs.sendMessage).toHaveBeenCalledWith(7, { kind: 'echo/notice', error: problem }, { frameId: 2 });
    },
  );

  test('a shortcut starts the same journey in the frame it was pressed in', async () => {
    const { value, calls } = deps();
    await launchStyle('elaborate', 5, tab, value);
    expect(calls).toEqual(['inject:5', 'prepare', 'badge:', 'echo/start-job:5']);
    await launchStyle('elaborate', 0, undefined, value);
    expect(calls).toHaveLength(4);
  });

  test('clicks on the parent item or other menus are ignored', async () => {
    const { value, calls } = deps();
    await launchRewrite({ menuItemId: 'echomebetter' }, tab, value);
    await launchRewrite({ menuItemId: 'someone-else:professional' }, tab, value);
    expect(calls).toEqual([]);
  });
});

describe('launchStyleMenu', () => {
  test('brings the foreground into the frame that was held in and opens the menu there, without waking the model yet', async () => {
    const { value, calls } = deps();
    await launchStyleMenu({ x: 12, y: 34 }, 4, tab, value);
    expect(calls).toEqual(['inject:4', 'echo/style-menu:4']);
    expect(value.tabs.sendMessage).toHaveBeenCalledWith(7, { kind: 'echo/style-menu', point: { x: 12, y: 34 } }, { frameId: 4 });
  });

  test("on the extension's own pages the foreground is already there", async () => {
    const { value, calls } = deps();
    await launchStyleMenu({ x: 1, y: 1 }, 0, { id: 7, url: 'chrome-extension://abc/ui/welcome/welcome.html' } as chrome.tabs.Tab, value);
    expect(calls).toEqual(['echo/style-menu:0']);
  });

  test('a frame that cannot be reached gets no menu', async () => {
    const { value, calls } = deps({ injectFails: () => true });
    await launchStyleMenu({ x: 1, y: 1 }, 0, tab, value);
    expect(calls).toEqual(['inject:0', 'badge:!']);
  });
});
