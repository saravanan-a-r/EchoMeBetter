import { UnsupportedTokenizerError, type MetaspaceJson, type PrependScheme, type PreTokenizerJson, type Split } from './types';

/** Takes one split of text, returns the words the model tokenizes independently. */
export type PreTokenizer = (split: Split) => Split[];

export function prependSchemeOf(json: MetaspaceJson): PrependScheme {
  if (json.prepend_scheme) return json.prepend_scheme;
  return json.add_prefix_space === false ? 'never' : 'always';
}

export function buildPreTokenizer(json: PreTokenizerJson | null): PreTokenizer | null {
  if (json === null) return null;
  switch (json.type) {
    case 'Metaspace':
      return metaspace(json.replacement, prependSchemeOf(json), json.split ?? true);
    case 'WhitespaceSplit':
      return whitespaceSplit;
    case 'Sequence': {
      const steps = json.pretokenizers.map(buildPreTokenizer).filter((step): step is PreTokenizer => step !== null);
      return (split) => steps.reduce<Split[]>((splits, step) => splits.flatMap(step), [split]);
    }
    default:
      throw new UnsupportedTokenizerError('pre_tokenizer', (json as { type: string }).type);
  }
}

/**
 * Spaces become the replacement character (normally "▁"), a leading one is
 * added according to the prepend scheme, and -- when `split` is on -- the
 * text is cut *before* every replacement character, so each word carries
 * its own leading "▁" (HuggingFace's `MergedWithNext`).
 */
function metaspace(replacement: string, scheme: PrependScheme, split: boolean): PreTokenizer {
  return ({ text, originalStart }) => {
    let replaced = text.replaceAll(' ', replacement);
    const prepend = scheme === 'always' || (scheme === 'first' && originalStart === 0);
    if (prepend && !replaced.startsWith(replacement)) replaced = replacement + replaced;
    if (!split) return replaced.length > 0 ? [{ text: replaced, originalStart }] : [];

    const words: Split[] = [];
    let current = '';
    for (const char of replaced) {
      if (char === replacement && current.length > 0) {
        words.push({ text: current, originalStart });
        current = '';
      }
      current += char;
    }
    if (current.length > 0) words.push({ text: current, originalStart });
    return words;
  };
}

/** Split on Unicode whitespace, dropping it. */
function whitespaceSplit({ text, originalStart }: Split): Split[] {
  return text
    .split(/\s+/u)
    .filter((word) => word.length > 0)
    .map((word) => ({ text: word, originalStart }));
}
