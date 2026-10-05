/**
 * Two builds, one `webpack` run:
 *
 *   extension   service worker, popup, welcome page, offscreen document and
 *               the inference worker, as ES modules (MV3 runs a module service
 *               worker; the worker loads onnxruntime-web's WebAssembly glue
 *               with a dynamic import).
 *   foreground  the script injected into pages on demand. chrome.scripting
 *               injects classic scripts, so this one is a self-contained IIFE.
 *
 * Loaded through @babel/register (see babel.config.cjs).
 *
 * The model is not bundled: the extension downloads it when the user asks
 * (see model.source.json).
 */
import CopyWebpackPlugin from 'copy-webpack-plugin';
import HtmlWebpackPlugin from 'html-webpack-plugin';
import MiniCssExtractPlugin from 'mini-css-extract-plugin';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Configuration, RuleSetRule } from 'webpack';
import { ExtensionAssetsPlugin } from './tooling/extensionAssets';

const root = __dirname;
const dist = join(root, 'dist');
const { version } = require('./package.json') as { version: string };

const PAGES = {
  popup: { entry: './src/ui/popup/main.tsx', html: 'ui/popup/popup.html' },
  welcome: { entry: './src/ui/welcome/main.tsx', html: 'ui/welcome/welcome.html' },
  offscreen: { entry: './src/offscreen/offscreen.ts', html: 'offscreen/offscreen.html' },
} as const;

const scripts: RuleSetRule = { test: /\.tsx?$/, exclude: /node_modules/, use: 'babel-loader' };
const inlineStylesheet: RuleSetRule = { test: /\.css$/, resourceQuery: /inline/, type: 'asset/source', use: ['postcss-loader'] };

function shared(production: boolean): Configuration {
  return {
    mode: production ? 'production' : 'development',
    context: root,
    // eval-based source maps are forbidden by the extension CSP.
    devtool: production ? false : 'cheap-module-source-map',
    resolve: { extensions: ['.ts', '.tsx', '.js', '.mjs'] },
    performance: {
      maxEntrypointSize: 512 * 1024,
      maxAssetSize: 512 * 1024,
      // The budget is for JavaScript and CSS; the WebAssembly runtime is a fixed vendor binary.
      assetFilter: (asset: string) => /\.(js|css)$/.test(asset),
    },
  };
}

export default (_env: unknown, argv: { mode?: string }): Configuration[] => {
  const production = argv.mode !== 'development';

  const extension: Configuration = {
    ...shared(production),
    name: 'extension',
    target: ['web', 'es2022'],
    entry: {
      background: './src/background/index.ts',
      ...Object.fromEntries(Object.entries(PAGES).map(([name, page]) => [name, page.entry])),
    },
    output: {
      path: dist,
      module: true,
      publicPath: '/',
      filename: ({ chunk }) => (chunk?.name === 'background' ? 'background.js' : 'assets/[name].[contenthash:8].js'),
      chunkFilename: 'assets/[name].[contenthash:8].js',
      // The foreground build writes foreground.js into the same folder after this one.
      clean: { keep: /^foreground\.js/ },
    },
    resolve: {
      ...shared(production).resolve,
      // onnxruntime-web's build that loads its WebAssembly glue at run time
      // from env.wasm.wasmPaths (dist/ort/), so its threads start from that
      // file rather than from our bundle.
      conditionNames: ['onnxruntime-web-use-extern-wasm', 'browser', 'import', 'module', 'default'],
    },
    module: {
      rules: [
        scripts,
        {
          // onnxruntime-web reads import.meta.url at run time to locate
          // itself; keep it native instead of baking in a build-machine path.
          test: /[\\/]node_modules[\\/]onnxruntime-web[\\/]/,
          parser: { importMeta: { url: false } },
        },
        { test: /\.css$/, oneOf: [inlineStylesheet, { use: [MiniCssExtractPlugin.loader, 'css-loader', 'postcss-loader'] }] },
      ],
    },
    optimization: {
      // The service worker must be one self-contained file; pages share chunks.
      splitChunks: { chunks: (chunk) => chunk.name !== 'background' },
    },
    plugins: [
      new MiniCssExtractPlugin({ filename: 'assets/[name].[contenthash:8].css' }),
      ...Object.entries(PAGES).map(
        ([name, page]) =>
          new HtmlWebpackPlugin({
            // Plain HTML, used as-is: no template engine runs over it.
            templateContent: readFileSync(join(root, 'src', page.html), 'utf8'),
            filename: page.html,
            chunks: [name],
            scriptLoading: 'module',
          }),
      ),
      new CopyWebpackPlugin({
        patterns: [
          { from: 'public', to: '.' },
          ...['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'].map((file) => ({
            from: `node_modules/onnxruntime-web/dist/${file}`,
            to: `ort/${file}`,
            // 14 MB of WebAssembly that is already optimised; skip minifier passes.
            info: { minimized: true },
          })),
        ],
      }),
      new ExtensionAssetsPlugin({ version }),
    ],
  };

  const foreground: Configuration = {
    ...shared(production),
    name: 'foreground',
    dependencies: ['extension'],
    target: ['web', 'es2022'],
    entry: { foreground: './src/foreground/index.ts' },
    output: { path: dist, filename: '[name].js', iife: true, clean: false },
    module: { rules: [scripts, inlineStylesheet] },
    optimization: { splitChunks: false, runtimeChunk: false },
  };

  return [extension, foreground];
};
