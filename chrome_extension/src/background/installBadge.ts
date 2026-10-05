/**
 * The toolbar icon says when the model still has to be downloaded: a "!"
 * badge and a tooltip, until the download completes. The popup behind the
 * icon is where the download starts.
 *
 * This is the icon-wide badge; the per-tab badge for "can't edit text on
 * this page" (rewriteLauncher.ts) takes precedence on its own tab.
 */
import { modelFrom, type InstalledModelRecord } from '../shared/modelInstall';

export type BadgeApi = Pick<typeof chrome.action, 'setBadgeText' | 'setBadgeBackgroundColor' | 'setTitle'>;

export async function showInstallState(installed: InstalledModelRecord | null, sourceUrl: string, action: BadgeApi = chrome.action): Promise<void> {
  if (modelFrom(installed, sourceUrl)) {
    await action.setBadgeText({ text: '' });
    await action.setTitle({ title: 'EchoMeBetter' });
    return;
  }
  await action.setBadgeBackgroundColor({ color: '#6236F5' });
  await action.setBadgeText({ text: '!' });
  await action.setTitle({ title: 'EchoMeBetter: download the writing model to start' });
}
