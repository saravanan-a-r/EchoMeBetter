import { describe, expect, jest, test } from '@jest/globals';
import { JobRouter, type PortLike } from '../../../background/jobRouter';
import type { JobEvent, OffscreenCommand, WokeFromRest } from '../../../shared/messages';

function fakePort() {
  const sent: JobEvent[] = [];
  let onMessage!: (message: unknown) => void;
  let onDisconnect!: () => void;
  const port: PortLike = {
    name: 'echo/job',
    postMessage: (message) => sent.push(message),
    onMessage: { addListener: (listener) => (onMessage = listener) },
    onDisconnect: { addListener: (listener) => (onDisconnect = listener) },
  };
  return { port, sent, send: (message: unknown) => onMessage(message), disconnect: () => onDisconnect() };
}

function router(ensureEngineHost: () => Promise<void> = async () => undefined, wakingFromRest: () => Promise<WokeFromRest | null> = async () => null) {
  const commands: OffscreenCommand[] = [];
  const onActivity = jest.fn();
  const instance = new JobRouter({ ensureEngineHost, sendToEngine: async (command) => void commands.push(command), onActivity, wakingFromRest });
  return { instance, commands, onActivity };
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('JobRouter', () => {
  test('starts the engine host, forwards the job, and routes events back to the owning port only', async () => {
    const { instance, commands, onActivity } = router();
    const a = fakePort();
    const b = fakePort();
    instance.handlePort(a.port);
    instance.handlePort(b.port);
    a.send({ kind: 'job/request', jobId: 'A', style: 'concise', text: 'one' });
    b.send({ kind: 'job/request', jobId: 'B', style: 'friendly', text: 'two' });
    await flush();
    expect(commands).toEqual([
      { target: 'offscreen', kind: 'engine/rewrite', jobId: 'A', style: 'concise', text: 'one' },
      { target: 'offscreen', kind: 'engine/rewrite', jobId: 'B', style: 'friendly', text: 'two' },
    ]);

    instance.handleEngineEvent({ type: 'job-phase', jobId: 'B', phase: 'rewriting' });
    instance.handleEngineEvent({ type: 'job-done', jobId: 'A', text: 'ONE' });
    expect(a.sent).toEqual([{ kind: 'job/done', jobId: 'A', text: 'ONE' }]);
    expect(b.sent).toEqual([{ kind: 'job/phase', jobId: 'B', phase: 'rewriting', progress: undefined }]);
    expect(instance.activeJobCount).toBe(1);
    expect(onActivity).toHaveBeenCalled();
  });

  test('a page going away cancels its running job', async () => {
    const { instance, commands } = router();
    const page = fakePort();
    instance.handlePort(page.port);
    page.send({ kind: 'job/request', jobId: 'A', style: 'concise', text: 'x' });
    await flush();
    page.disconnect();
    await flush();
    expect(commands.at(-1)).toEqual({ target: 'offscreen', kind: 'engine/cancel', jobId: 'A' });
    expect(instance.activeJobCount).toBe(0);
  });

  test('cancelling while the engine host starts means the rewrite is never sent', async () => {
    let release!: () => void;
    const { instance, commands } = router(() => new Promise<void>((resolve) => (release = resolve)));
    const page = fakePort();
    instance.handlePort(page.port);
    page.send({ kind: 'job/request', jobId: 'A', style: 'concise', text: 'x' });
    page.send({ kind: 'job/cancel', jobId: 'A' });
    release();
    await flush();
    expect(commands).toEqual([{ target: 'offscreen', kind: 'engine/cancel', jobId: 'A' }]);
  });

  test('an engine host that cannot start fails the job instead of leaving the page waiting', async () => {
    const { instance } = router(async () => {
      throw new Error('offscreen blocked');
    });
    const page = fakePort();
    instance.handlePort(page.port);
    page.send({ kind: 'job/request', jobId: 'A', style: 'concise', text: 'x' });
    await flush();
    expect(page.sent).toEqual([{ kind: 'job/failed', jobId: 'A', error: { code: 'MODEL_LOAD_FAILED', details: { message: 'offscreen blocked' } } }]);
  });

  test('malformed or foreign messages are ignored', async () => {
    const { instance, commands } = router();
    const page = fakePort();
    instance.handlePort(page.port);
    page.send({ kind: 'job/request', jobId: 'A', style: 'shouting', text: 'x' });
    page.send('hello');
    await flush();
    expect(commands).toEqual([]);
  });

  test('a job that wakes the model from an idle rest says so when it is done; the next one does not', async () => {
    let resting = true;
    const { instance } = router(async () => undefined, async () => (resting ? { idleMinutes: 15 } : null));
    const page = fakePort();
    instance.handlePort(page.port);
    page.send({ kind: 'job/request', jobId: 'A', style: 'concise', text: 'one' });
    await flush();
    resting = false; // the model loaded for A
    instance.handleEngineEvent({ type: 'job-done', jobId: 'A', text: 'ONE' });
    page.send({ kind: 'job/request', jobId: 'B', style: 'concise', text: 'two' });
    await flush();
    instance.handleEngineEvent({ type: 'job-done', jobId: 'B', text: 'TWO' });
    expect(page.sent).toEqual([
      { kind: 'job/done', jobId: 'A', text: 'ONE', wokeFromRest: { idleMinutes: 15 } },
      { kind: 'job/done', jobId: 'B', text: 'TWO' },
    ]);
  });
});
