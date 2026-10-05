/** @type {import('jest').Config} */
module.exports = {
  roots: ['<rootDir>/src'],
  testMatch: ['<rootDir>/src/test/**/*.test.{ts,tsx}'],
  // Per-file `@jest-environment jsdom` docblocks opt DOM tests in.
  testEnvironment: 'node',
  transform: { '^.+\\.[jt]sx?$': 'babel-jest' },
  moduleNameMapper: {
    // Stylesheets are a build concern; tests get the CSS text as an empty string.
    '\\.css\\?inline$': '<rootDir>/src/test/helpers/cssStub.cjs',
    '\\.css$': '<rootDir>/src/test/helpers/cssStub.cjs',
  },
  setupFilesAfterEnv: ['<rootDir>/src/test/helpers/setup.ts'],
  restoreMocks: true,
  testTimeout: 30000,
};
