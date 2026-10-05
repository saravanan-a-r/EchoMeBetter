/** A stylesheet compiled at build time and imported as a string (webpack `?inline` rule). */
declare module '*.css?inline' {
  const css: string;
  export default css;
}

declare module '*.css';
