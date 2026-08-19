---
title: Writing blogs in markdown
date: 2026-08-20
---

# Writing blogs in markdown

Previously, web files for this blog were written in emacs, editing the HTML by hand. Now they are written in markdown.
That makes life a lot easier. 
The key piece of html - suggested by claude - is:
```
<div id="rendered" style="max-width: 720px; padding: 2rem;"></div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex/dist/contrib/auto-render.min.js"></script>
<script>
 fetch('./markdown-posts/blogs-in-markdown.md')
   .then(r => r.text())
   .then(md => {
     // strip YAML front matter (--- ... ---) added for the RSS feed,
     // so it doesn't render as visible content
     md = md.replace(/^---[\s\S]*?---\s*/, '');
     document.getElementById('rendered').innerHTML = marked.parse(md);
     renderMathInElement(document.getElementById('rendered'), {
       delimiters: [
         { left: '$$', right: '$$', display: true },
         { left: '$',  right: '$',  display: false }
       ]
     });
   });
</script>
```
