/**
 * The rewrite styles the extension offers, in menu order.
 *
 * This is the product-facing catalogue: ids, labels and one-line hints. How a
 * style is *asked of the model* lives in the model's own manifest
 * (`model.json` → `styles`), so a new model can change its prompts without
 * touching this list, and a manifest that does not cover every id here is
 * rejected when the model loads.
 */
export const STYLE_IDS = ['professional', 'grammar', 'friendly', 'concise', 'elaborate'] as const;

export type StyleId = (typeof STYLE_IDS)[number];

export interface StyleDescriptor {
  readonly id: StyleId;
  readonly label: string;
  readonly hint: string;
}

export const STYLES: readonly StyleDescriptor[] = [
  { id: 'professional', label: 'Professional', hint: 'Polished and workplace-ready' },
  { id: 'grammar', label: 'Grammar', hint: 'Fix grammar and spelling only' },
  { id: 'friendly', label: 'Friendly', hint: 'Warm and approachable' },
  { id: 'concise', label: 'Concise', hint: 'Shorter, same meaning' },
  { id: 'elaborate', label: 'Elaborate', hint: 'Fuller, with more detail' },
];

export function isStyleId(value: unknown): value is StyleId {
  return typeof value === 'string' && (STYLE_IDS as readonly string[]).includes(value);
}

export function styleLabel(id: StyleId): string {
  const style = STYLES.find((candidate) => candidate.id === id);
  if (!style) throw new Error(`unknown style: ${id}`);
  return style.label;
}
