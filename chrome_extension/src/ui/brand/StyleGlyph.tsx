/** A small line icon per rewrite style, drawn on a 20×20 grid with the current text colour. */
import type { StyleId } from '../../shared/styles';

const PATHS: Record<StyleId, string> = {
  // briefcase
  professional: 'M7 6V4.8C7 4.36 7.36 4 7.8 4h4.4c.44 0 .8.36.8.8V6M3.5 6.5h13v8.7c0 .44-.36.8-.8.8H4.3a.8.8 0 0 1-.8-.8V6.5Zm0 4h13',
  // check in a speech line
  grammar: 'M4 5.5h12M4 9.5h7M4 13.5h5M11.5 13.8l2 2 3.5-4.3',
  // smile
  friendly: 'M10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm-3-5.5c.8 1.2 1.8 1.8 3 1.8s2.2-.6 3-1.8M7.5 8h.01M12.5 8h.01',
  // arrows in
  concise: 'M3.5 10h5m0 0L6 7.5M8.5 10 6 12.5M16.5 10h-5m0 0L14 7.5M11.5 10l2.5 2.5',
  // arrows out
  elaborate: 'M8.5 10h-5m0 0L6 7.5M3.5 10 6 12.5M11.5 10h5m0 0L14 7.5M16.5 10 14 12.5',
};

export function StyleGlyph({ style, size = 18 }: { style: StyleId; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <path d={PATHS[style]} stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
