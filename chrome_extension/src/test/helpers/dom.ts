/**
 * jsdom does not implement `isContentEditable`; browsers do. This gives jsdom
 * the browser's behaviour so contenteditable code paths can be tested.
 */
export function installContentEditableSupport(): void {
  if ('isContentEditable' in HTMLElement.prototype) return;
  Object.defineProperty(HTMLElement.prototype, 'isContentEditable', {
    configurable: true,
    get(this: HTMLElement) {
      const host = this.closest('[contenteditable]');
      return host !== null && host.getAttribute('contenteditable') !== 'false';
    },
  });
}

export function selectContents(node: Node): void {
  const range = document.createRange();
  range.selectNodeContents(node);
  const selection = document.getSelection()!;
  selection.removeAllRanges();
  selection.addRange(range);
}
