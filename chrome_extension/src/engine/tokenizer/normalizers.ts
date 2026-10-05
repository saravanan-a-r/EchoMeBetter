import { PrecompiledCharsMap } from './precompiled';
import { compilePattern, UnsupportedTokenizerError, type NormalizerJson } from './types';

export type Normalizer = (text: string) => string;

export function buildNormalizer(json: NormalizerJson | null): Normalizer | null {
  if (json === null) return null;
  switch (json.type) {
    case 'Sequence': {
      const steps = json.normalizers.map(buildNormalizer).filter((step): step is Normalizer => step !== null);
      return (text) => steps.reduce((current, step) => step(current), text);
    }
    case 'Precompiled': {
      if (!json.precompiled_charsmap) return null;
      const map = new PrecompiledCharsMap(json.precompiled_charsmap);
      return (text) => map.normalize(text);
    }
    case 'Replace': {
      const pattern = compilePattern(json.pattern);
      return (text) => text.replace(pattern, () => json.content);
    }
    case 'NFC':
    case 'NFD':
    case 'NFKC':
    case 'NFKD': {
      const form = json.type;
      return (text) => text.normalize(form);
    }
    case 'Lowercase':
      return (text) => text.toLowerCase();
    case 'Strip':
      return (text) => {
        let result = text;
        if (json.strip_left) result = result.replace(/^\s+/u, '');
        if (json.strip_right) result = result.replace(/\s+$/u, '');
        return result;
      };
    case 'Prepend':
      return (text) => (text.length > 0 ? json.prepend + text : text);
    default:
      throw new UnsupportedTokenizerError('normalizer', (json as { type: string }).type);
  }
}
