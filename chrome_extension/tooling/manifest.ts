/**
 * manifest.json, generated at build time so its paths always match the
 * files the build actually emits.
 */
export function createManifest(version: string): chrome.runtime.ManifestV3 {
  const icons = { 16: 'icons/icon-16.png', 32: 'icons/icon-32.png', 48: 'icons/icon-48.png', 128: 'icons/icon-128.png' };
  return {
    manifest_version: 3,
    name: 'EchoMeBetter',
    version,
    description:
      'Rewrite selected text in any text box: professional, grammar, friendly, concise or elaborate. Runs entirely on your device.',
    // runtime.getContexts (offscreen lookup) arrived in Chrome 116.
    minimum_chrome_version: '116',
    icons,
    action: { default_title: 'EchoMeBetter', default_popup: 'ui/popup/popup.html', default_icon: icons },
    background: { service_worker: 'background.js', type: 'module' },
    // No required host permissions: a context-menu click grants activeTab for that
    // tab, which is all the foreground injection needs.
    permissions: ['contextMenus', 'activeTab', 'scripting', 'offscreen', 'storage', 'alarms'],
    // Keyboard shortcuts listen on web pages, which needs access to them. It
    // is asked for from the popup when the user turns shortcuts on, never at
    // install, and handed back when they turn shortcuts off.
    optional_host_permissions: ['<all_urls>'],
    content_security_policy: {
      // onnxruntime-web compiles WebAssembly; nothing else is relaxed.
      extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
    },
    // Cross-origin isolation gives extension pages SharedArrayBuffer, which
    // onnxruntime-web needs to run on more than one thread.
    cross_origin_embedder_policy: { value: 'require-corp' },
    cross_origin_opener_policy: { value: 'same-origin' },
  } as chrome.runtime.ManifestV3;
}
