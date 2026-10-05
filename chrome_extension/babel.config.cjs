/**
 * One TypeScript/JSX transform for everything: webpack (babel-loader), Jest
 * (babel-jest) and webpack's own TypeScript config (@babel/register).
 * Type checking is `npm run typecheck`; Babel only strips types.
 */
module.exports = (api) => {
  const bundling = api.caller((caller) => caller?.name === 'babel-loader');
  return {
    presets: [
      [
        '@babel/preset-env',
        bundling
          ? { targets: { chrome: '116' }, modules: false, bugfixes: true }
          : { targets: { node: 'current' } },
      ],
      ['@babel/preset-react', { runtime: 'automatic' }],
      ['@babel/preset-typescript', { onlyRemoveTypeImports: true }],
    ],
  };
};
