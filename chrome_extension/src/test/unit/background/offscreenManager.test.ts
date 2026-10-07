import { describe, expect, jest, test } from '@jest/globals';
import { OffscreenManager, type OffscreenApis } from '../../../background/offscreenManager';
import { DEFAULT_COMPUTE } from '../../../shared/compute';

function apis(initiallyOpen = false) {
  let open = initiallyOpen;
  const events: string[] = [];
  const createDocument = jest.fn(async (_parameters: chrome.offscreen.CreateParameters) => {
    await new Promise((resolve) => setTimeout(resolve, 5));
    open = true;
    events.push('create');
  });
  const closeDocument = jest.fn(async () => {
    await new Promise((resolve) => setTimeout(resolve, 5));
    open = false;
    events.push('close');
  });
  const sendMessage = jest.fn(async (_message: unknown): Promise<unknown> => ({ ok: true }));
  const value = {
    offscreen: { createDocument, closeDocument },
    runtime: { getContexts: async () => (open ? [{}] : []), getURL: (path: string) => `chrome-extension://id/${path}`, sendMessage },
  } as unknown as OffscreenApis;
  return { value, createDocument, closeDocument, sendMessage, events };
}

const compute = async () => DEFAULT_COMPUTE;

describe('OffscreenManager', () => {
  test('concurrent ensure() calls create exactly one document, then start its worker with the compute settings', async () => {
    const { value, createDocument, sendMessage } = apis();
    const manager = new OffscreenManager(compute, value);
    await Promise.all([manager.ensure(), manager.ensure(), manager.ensure()]);
    expect(createDocument).toHaveBeenCalledTimes(1);
    expect(createDocument.mock.calls[0]).toEqual([expect.objectContaining({ url: 'offscreen/offscreen.html', reasons: ['WORKERS'] })]);
    expect(sendMessage.mock.calls).toEqual([[{ target: 'offscreen', kind: 'engine/start', compute: DEFAULT_COMPUTE }]]);
  });

  test('an existing document (service worker restarted) is reused', async () => {
    const { value, createDocument, sendMessage } = apis(true);
    await new OffscreenManager(compute, value).ensure();
    expect(createDocument).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  test('a document whose worker could not be started is closed again, so the next ensure() starts afresh', async () => {
    const { value, sendMessage, createDocument, closeDocument } = apis();
    sendMessage.mockRejectedValueOnce(new Error('Receiving end does not exist.'));
    const manager = new OffscreenManager(compute, value);
    await expect(manager.ensure()).rejects.toThrow('Receiving end does not exist.');
    expect(closeDocument).toHaveBeenCalledTimes(1);
    await manager.ensure();
    expect(createDocument).toHaveBeenCalledTimes(2);
  });

  test('close() only closes a document that exists', async () => {
    const closed = apis(false);
    await new OffscreenManager(compute, closed.value).close();
    expect(closed.closeDocument).not.toHaveBeenCalled();
    const open = apis(true);
    await new OffscreenManager(compute, open.value).close();
    expect(open.closeDocument).toHaveBeenCalledTimes(1);
  });

  test('work asked for while the document closes goes to a new document, not the closing one', async () => {
    const { value, events } = apis(true);
    const manager = new OffscreenManager(compute, value);
    const closing = manager.close();
    await manager.ensure();
    await closing;
    expect(events).toEqual(['close', 'create']);
    expect(await manager.exists()).toBe(true);
  });
});
