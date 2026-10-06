import { describe, expect, jest, test } from '@jest/globals';
import { OffscreenManager, type OffscreenApis } from '../../../background/offscreenManager';

function apis(initiallyOpen = false) {
  let open = initiallyOpen;
  const createDocument = jest.fn(async (_parameters: chrome.offscreen.CreateParameters) => {
    await new Promise((resolve) => setTimeout(resolve, 5));
    open = true;
  });
  const closeDocument = jest.fn(async () => {
    open = false;
  });
  const sendMessage = jest.fn(async (_message: unknown) => ({ ok: true }));
  const value = {
    offscreen: { createDocument, closeDocument },
    runtime: { getContexts: async () => (open ? [{}] : []), getURL: (path: string) => `chrome-extension://id/${path}`, sendMessage },
  } as unknown as OffscreenApis;
  return { value, createDocument, closeDocument, sendMessage };
}

describe('OffscreenManager', () => {
  test('concurrent ensure() calls create exactly one document, then ping it', async () => {
    const { value, createDocument, sendMessage } = apis();
    const manager = new OffscreenManager(value);
    await Promise.all([manager.ensure(), manager.ensure(), manager.ensure()]);
    expect(createDocument).toHaveBeenCalledTimes(1);
    expect(createDocument.mock.calls[0]).toEqual([expect.objectContaining({ url: 'offscreen/offscreen.html', reasons: ['WORKERS'] })]);
    expect(sendMessage).toHaveBeenCalledWith({ target: 'offscreen', kind: 'engine/ping' });
  });

  test('an existing document (service worker restarted) is reused', async () => {
    const { value, createDocument } = apis(true);
    await new OffscreenManager(value).ensure();
    expect(createDocument).not.toHaveBeenCalled();
  });

  test('close() only closes a document that exists', async () => {
    const closed = apis(false);
    await new OffscreenManager(closed.value).close();
    expect(closed.closeDocument).not.toHaveBeenCalled();
    const open = apis(true);
    await new OffscreenManager(open.value).close();
    expect(open.closeDocument).toHaveBeenCalledTimes(1);
  });
});
