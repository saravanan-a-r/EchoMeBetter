/**
 * Webpack plugin: manifest.json, emitted as a webpack asset from
 * tooling/manifest.ts so its paths always match the files the build emits.
 *
 * The model is not part of the package: the extension downloads it at the
 * user's request (see model.source.json).
 */
import type { Compiler } from 'webpack';
import { createManifest } from './manifest';

const PLUGIN = 'EchoMeBetterExtensionAssets';

export class ExtensionAssetsPlugin {
  constructor(private readonly options: { version: string }) {}

  apply(compiler: Compiler): void {
    const { RawSource } = compiler.webpack.sources;

    compiler.hooks.thisCompilation.tap(PLUGIN, (compilation) => {
      compilation.hooks.processAssets.tap({ name: PLUGIN, stage: compiler.webpack.Compilation.PROCESS_ASSETS_STAGE_ADDITIONAL }, () => {
        compilation.emitAsset('manifest.json', new RawSource(`${JSON.stringify(createManifest(this.options.version), null, 2)}\n`));
      });
    });
  }
}
