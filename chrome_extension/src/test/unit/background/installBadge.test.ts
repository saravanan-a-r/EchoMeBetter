import { describe, expect, jest, test } from '@jest/globals';
import { showInstallState, type BadgeApi } from '../../../background/installBadge';
import { installedRecord } from '../../helpers/fixtures';

function action() {
  return {
    setBadgeText: jest.fn(async (_details: { text: string }) => undefined),
    setBadgeBackgroundColor: jest.fn(async () => undefined),
    setTitle: jest.fn(async (_details: { title: string }) => undefined),
  };
}

const SOURCE = 'https://models.example.test/m/';
const record = installedRecord(SOURCE);

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

  test('a model without any style still has nothing to rewrite with', async () => {
    const api = action();
    await showInstallState(installedRecord(SOURCE, []), SOURCE, api as unknown as BadgeApi);
    expect(api.setBadgeText).toHaveBeenCalledWith({ text: '!' });
    expect(api.setTitle).toHaveBeenCalledWith({ title: 'EchoMeBetter: download a style to start' });
  });

  test('clears the flag once the model and a style are downloaded', async () => {
    const api = action();
    await showInstallState(record, SOURCE, api as unknown as BadgeApi);
    expect(api.setBadgeText).toHaveBeenCalledWith({ text: '' });
    expect(api.setTitle).toHaveBeenCalledWith({ title: 'EchoMeBetter' });
  });
});
