/** Types for tokens.cjs: the design tokens in tokens.json, resolved. */
import type source from './tokens.json';

type Names<Group> = Exclude<keyof Group, `$${string}`>;
type Scale<Group> = { readonly [Step in Names<Group>]: string };

export type ThemeName = Names<(typeof source)['theme']>;
/** Every theme has every token of the light theme. */
export type ThemeToken = Names<(typeof source)['theme']['light']>;

export declare const palette: {
  readonly [Hue in Names<(typeof source)['palette']>]: (typeof source)['palette'][Hue] extends { $value: string } ? string : Scale<(typeof source)['palette'][Hue]>;
};
export declare const themes: { readonly [Theme in ThemeName]: { readonly [Token in ThemeToken]: string } };
export declare const gradient: Scale<(typeof source)['gradient']>;
export declare const brand: Scale<(typeof source)['brand']>;
export declare const font: { readonly [Family in Names<(typeof source)['font']>]: readonly string[] };
export declare const shadow: Scale<(typeof source)['shadow']>;
export declare const motion: {
  readonly duration: Scale<(typeof source)['motion']['duration']>;
  readonly easing: { readonly [Name in Names<(typeof source)['motion']['easing']>]: readonly number[] };
};
/** A hex colour as CSS rgb() channels. */
export declare function channels(hex: string): string;
