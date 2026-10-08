/** Types for tailwind.cjs, for the tests that check it. */
import type { Config } from 'tailwindcss';

export declare const theme: { extend: Record<string, Record<string, unknown>> };
export declare function themeVariables(options: { selector: string; scheme: 'system' | 'light' | 'dark' | 'inverse' }): NonNullable<Config['plugins']>[number];
