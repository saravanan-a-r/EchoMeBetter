/**
 * Reversible text transforms a model can ask for in its manifest
 * (`inputTransforms`): applied to the user's text before tokenizing, undone
 * on the model's output, last-applied first.
 */
import type { TextTransformName } from '../manifest';

/**
 * EchoMeBetter's SentencePiece decoder turns every "▁" into a space, so a
 * literal "▁" in the user's text would not survive a round trip. The text is
 * escaped into Private Use Area code points first, exactly as the training
 * pipeline does (tokenizer/training/marker_escape.py): one left-to-right
 * pass, escaping the escape character too, so the scheme is lossless for
 * any input.
 */
const SPIECE_UNDERLINE = '▁';
const ESC = '';
const ESCAPED_UNDERLINE = '';

export function escapeSpieceMarkers(text: string): string {
  if (!text.includes(ESC) && !text.includes(SPIECE_UNDERLINE)) return text;
  let out = '';
  for (const char of text) {
    if (char === ESC) out += ESC + ESC;
    else if (char === SPIECE_UNDERLINE) out += ESC + ESCAPED_UNDERLINE;
    else out += char;
  }
  return out;
}

export function unescapeSpieceMarkers(text: string): string {
  if (!text.includes(ESC)) return text;
  let out = '';
  for (let i = 0; i < text.length; i++) {
    const char = text[i]!;
    const next = text[i + 1];
    if (char === ESC && (next === ESC || next === ESCAPED_UNDERLINE)) {
      out += next === ESC ? ESC : SPIECE_UNDERLINE;
      i++;
    } else {
      out += char;
    }
  }
  return out;
}

const TRANSFORMS: Record<TextTransformName, { readonly apply: (text: string) => string; readonly undo: (text: string) => string }> = {
  'escape-spiece-markers': { apply: escapeSpieceMarkers, undo: unescapeSpieceMarkers },
};

export function applyInputTransforms(text: string, names: readonly TextTransformName[]): string {
  return names.reduce((current, name) => TRANSFORMS[name].apply(current), text);
}

export function undoInputTransforms(text: string, names: readonly TextTransformName[]): string {
  return [...names].reverse().reduce((current, name) => TRANSFORMS[name].undo(current), text);
}
