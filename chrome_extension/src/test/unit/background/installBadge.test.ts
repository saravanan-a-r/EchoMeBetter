import { describe, expect, jest, test } from '@jest/globals';
import { showInstallState, type BadgeApi } from '../../../background/installBadge';

function action() {
  return {
    setBadgeText: jest.fn(async (_details: { text: string }) => undefined),
    setBadgeBackgroundColor: jest.fn(async () => undefined),
    setTitle: jest.fn(async (_details: { title: string }) => undefined),
  };
}

const SOURCE = 'https://models.example.test/m/';
const record = { sourceUrl: SOURCE, model: { id: 'x', displayName: 'X', placeholder: false, precision: 'int8', sizeBytes: 1 } };

describe('toolbar badge', () => {
  test('flags a missing model on the icon', async () => {
    const api = action();
    await showInstallState(null, SOURCE, api as unknown as BadgeApi);
    expect(api.setBadgeText).toHaveBeenCalledWith({ text: '!' });
    expect(api.setTitle).toHaveBeenCalledWith({ title: 'EchoMeBetter: download the writing model to start' });
  });

  test('a model from another URL still needs a download', async () => {
    const api = action();
    await showInstallState(record, 'https://models.example.test/other/', api as unknown as BadgeApi);
    expect(api.setBadgeText).toHaveBeenCalledWith({ text: '!' });
  });

  test('clears the flag once the model is downloaded', async () => {
    const api = action();
    await showInstallState(record, SOURCE, api as unknown as BadgeApi);
    expect(api.setBadgeText).toHaveBeenCalledWith({ text: '' });
    expect(api.setTitle).toHaveBeenCalledWith({ title: 'EchoMeBetter' });
  });
});
