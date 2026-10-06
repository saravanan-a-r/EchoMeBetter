/**
 * The toolbar icon says when nothing can be rewritten yet -- the model, or
 * every style's adapter, still has to be downloaded: a "!" badge and a
 * tooltip. The popup behind the icon is where downloads start.
 *
 * This is the icon-wide badge; the per-tab badge for "can't edit text on
 * this page" (rewriteLauncher.ts) takes precedence on its own tab.
 */
import { canRewrite, modelFrom, type InstalledModelRecord } from '../shared/modelInstall';

export type BadgeApi = Pick<typeof chrome.action, 'setBadgeText' | 'setBadgeBackgroundColor' | 'setTitle'>;

export async function showInstallState(installed: InstalledModelRecord | null, sourceUrl: string, action: BadgeApi = chrome.action): Promise<void> {
  const model = modelFrom(installed, sourceUrl);
  if (canRewrite(model)) {
    await action.setBadgeText({ text: '' });
    await action.setTitle({ title: 'EchoMeBetter' });
    return;
  }
  await action.setBadgeBackgroundColor({ color: '#6236F5' });
  await action.setBadgeText({ text: '!' });
  await action.setTitle({ title: model ? 'EchoMeBetter: download a style to start' : 'EchoMeBetter: download the writing model to start' });
}
