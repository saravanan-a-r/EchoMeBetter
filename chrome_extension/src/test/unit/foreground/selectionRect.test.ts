import { describe, expect, test } from '@jest/globals';
import { placeSelection, rectOf } from '../../../foreground/target/selectionRect';

// A 400x300 textarea at (50, 100). The selection measured in the invisible copy
// is its second line: 28px tall, starting 44px down and 10px in from the corner.
const element = rectOf(100, 50, 400, 450);
const secondLine = rectOf(44, 10, 72, 200);

describe('placing a selection inside a text control', () => {
  test('a selection near the top of a tall control stays near the top, not at the control\'s bottom', () => {
    const rect = placeSelection({ element, inCopy: secondLine, scrollTop: 0, scrollLeft: 0, singleLine: false });
    expect(rect).toMatchObject({ top: 144, bottom: 172, left: 60, right: 250 });
    expect(rect.bottom).toBeLessThan(element.bottom - 100);
  });

  test('follows the control as it scrolls', () => {
    const rect = placeSelection({ element, inCopy: secondLine, scrollTop: 30, scrollLeft: 0, singleLine: false });
    expect(rect).toMatchObject({ top: 114, bottom: 142 });
  });

  test('a selection half scrolled out is cut at the control\'s edge', () => {
    const rect = placeSelection({ element, inCopy: secondLine, scrollTop: 58, scrollLeft: 0, singleLine: false });
    expect(rect.top).toBe(100);
    expect(rect.bottom).toBe(114);
  });

  test('a selection scrolled out of view is pinned to the nearest edge', () => {
    const above = placeSelection({ element, inCopy: secondLine, scrollTop: 500, scrollLeft: 0, singleLine: false });
    expect(above).toMatchObject({ top: 100, bottom: 100 });
    const below = placeSelection({ element, inCopy: rectOf(900, 10, 928, 200), scrollTop: 0, scrollLeft: 0, singleLine: false });
    expect(below).toMatchObject({ top: 400, bottom: 400 });
  });

  test('a one-line input keeps its own height and takes the text\'s horizontal extent, scrolled', () => {
    const input = rectOf(500, 50, 540, 450);
    const rect = placeSelection({ element: input, inCopy: rectOf(0, 120, 20, 300), scrollTop: 0, scrollLeft: 40, singleLine: true });
    expect(rect).toMatchObject({ top: 500, bottom: 540, left: 130, right: 310 });
  });
});
