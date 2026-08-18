---
title: Your post title
date: YYYY-MM-DD
---

# Your post title

A one- or two-line intro, e.g. linking to the repo, gist, or demo data.

## First section

A paragraph goes here. Use [links](https://example.com) and `inline code` freely.

```python
# a fenced code block
print("hello")
```

```bash
# a shell example
curl -s https://example.com | head
```

## More

- bullet points
- and more bullets

<!-- Use absolute URLs for images and links so they resolve inside RSS
     readers (e.g. https://ljmartin.github.io/blog/pics/foo.png), not
     relative ones like ./pics/foo.png. -->

<!-- NOTE: the client-side renderer (marked.js) does NOT understand the
     YAML front matter above. Your post's HTML wrapper must strip it
     with `md = md.replace(/^---[\s\S]*?---\s*/, '')` before
     `marked.parse(md)`, otherwise the --- ... --- block renders as
     visible text at the top of the page. -->

![alt text](https://ljmartin.github.io/blog/pics/example.png)