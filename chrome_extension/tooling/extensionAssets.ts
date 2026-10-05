/**
 * Webpack plugin: the parts of the extension that are not compiled code.
 *
 *   manifest.json   emitted as a webpack asset, from tooling/manifest.ts
 *   model/          the model folder named in model.selection.json, verified first
 *
 * The model is checked against its own model.json -- schema, file sizes and
 * SHA-256 -- before it is copied, so a half-copied or mismatched model fails
 * the build instead of failing on a user's machine. The files are hundreds
 * of megabytes, so they are cloned after webpack has emitted (copy-on-write
 * where the file system supports it) instead of being read into the
 * compilation's memory as assets.
 */
import { createHash } from 'node:crypto';
import { constants, copyFileSync, createReadStream, existsSync, mkdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type { Compiler } from 'webpack';
import { parseModelManifest, type ModelManifest } from '../src/engine/manifest';
import { createManifest } from './manifest';

const PLUGIN = 'EchoMeBetterExtensionAssets';

function sha256(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256');
    createReadStream(path)
      .on('data', (chunk) => hash.update(chunk))
      .on('error', reject)
      .on('end', () => resolve(hash.digest('hex')));
  });
}

export async function verifyModelFolder(folder: string): Promise<ModelManifest> {
  const manifestPath = join(folder, 'model.json');
  if (!existsSync(manifestPath)) {
    throw new Error(`model folder ${folder} has no model.json. Export the model into it first, or change model.selection.json.`);
  }
  const manifest = parseModelManifest(JSON.parse(readFileSync(manifestPath, 'utf8')));
  for (const [role, file] of Object.entries(manifest.files)) {
    const path = join(folder, file.path);
    if (!existsSync(path)) throw new Error(`model ${role} file is missing: ${path}`);
    const size = statSync(path).size;
    if (size !== file.bytes) throw new Error(`model ${role} file ${path} is ${size} bytes; model.json says ${file.bytes}`);
    if ((await sha256(path)) !== file.sha256) throw new Error(`model ${role} file ${path} does not match the sha256 in model.json`);
  }
  return manifest;
}

export function selectedModelFolder(projectRoot: string): string {
  const selection = JSON.parse(readFileSync(join(projectRoot, 'model.selection.json'), 'utf8')) as { activeModel?: unknown };
  if (typeof selection.activeModel !== 'string' || selection.activeModel.length === 0) {
    throw new Error('model.selection.json must name an "activeModel" folder under models/');
  }
  return join(projectRoot, 'models', selection.activeModel);
}

export class ExtensionAssetsPlugin {
  constructor(private readonly options: { projectRoot: string; version: string }) {}

  apply(compiler: Compiler): void {
    const { RawSource } = compiler.webpack.sources;

    compiler.hooks.thisCompilation.tap(PLUGIN, (compilation) => {
      compilation.hooks.processAssets.tap({ name: PLUGIN, stage: compiler.webpack.Compilation.PROCESS_ASSETS_STAGE_ADDITIONAL }, () => {
        compilation.emitAsset('manifest.json', new RawSource(`${JSON.stringify(createManifest(this.options.version), null, 2)}\n`));
      });
    });

    compiler.hooks.afterEmit.tapPromise(PLUGIN, async (compilation) => {
      const source = selectedModelFolder(this.options.projectRoot);
      compilation.fileDependencies.add(join(this.options.projectRoot, 'model.selection.json'));
      const manifest = await verifyModelFolder(source);
      const target = join(compilation.outputOptions.path!, 'model');
      mkdirSync(target, { recursive: true });
      for (const path of ['model.json', ...Object.values(manifest.files).map((file) => file.path)]) {
        copyFileSync(join(source, path), join(target, path), constants.COPYFILE_FICLONE);
      }
    });
  }
}
