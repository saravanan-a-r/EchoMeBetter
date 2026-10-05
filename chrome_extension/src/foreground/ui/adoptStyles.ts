/**
 * Attach CSS to a document or shadow root in a way a page's Content Security
 * Policy cannot veto.
 *
 * A `<style>` element is subject to the page's `style-src`; a constructed
 * stylesheet (CSSOM) is not. Many sites ship `style-src` without
 * 'unsafe-inline', and the overlay must look right on them too. The <style>
 * fallback only exists for environments without constructable stylesheets.
 */
export function adoptStyles(target: Document | ShadowRoot, css: string): () => void {
  const doc = target instanceof Document ? target : target.ownerDocument;
  const view = doc.defaultView as (Window & typeof globalThis) | null;
  if (view && 'adoptedStyleSheets' in target && typeof view.CSSStyleSheet?.prototype.replaceSync === 'function') {
    try {
      const sheet = new view.CSSStyleSheet();
      sheet.replaceSync(css);
      target.adoptedStyleSheets = [...target.adoptedStyleSheets, sheet];
      return () => {
        target.adoptedStyleSheets = target.adoptedStyleSheets.filter((candidate) => candidate !== sheet);
      };
    } catch {
      // Fall through to a <style> element.
    }
  }
  const style = doc.createElement('style');
  style.textContent = css;
  (target instanceof Document ? (target.head ?? target.documentElement) : target).appendChild(style);
  return () => style.remove();
}
