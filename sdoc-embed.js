(function() {
  const script = document.currentScript;
  const iframe = document.createElement('iframe');
  iframe.src = script.dataset.src;
  iframe.width = '100%';
  iframe.height = script.dataset.height || '500';
  iframe.style.cssText = 'border:1px solid #ddd; border-radius:8px; display:block;';
  iframe.loading = 'lazy';
  iframe.title = script.dataset.title || 'Document';
  script.replaceWith(iframe);
})();
