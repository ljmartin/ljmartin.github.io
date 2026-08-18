# RSS Feed Implementation — Handoff Notes

Context and decisions from the session that built an RSS feed for this blog. Read this to resume with full context instead of re-investigating everything.

---

## Goal

Add an RSS feed to the hand-written blog (no Jekyll/SSG; each post is a hand-edited HTML file, hosted on GitHub Pages at `ljmartin.github.io`, default branch `main`).

## What we discovered (important prior state)

- A feed already existed at `blog/feed.xml`, created ~Aug 2023 and described in blog post `blog/15_rss.html` ("Creating an RSS feed from HTML in python"). It used a Python script (`rfeed` + `BeautifulSoup`) that parsed each post's `<main>` and `datePublished` meta tag.
- **That feed was stale** — it only contained posts up to #14 (Fresnel). Posts #16–#34 were missing.
- Why it broke:
  1. Several posts had broken `datePublished` meta tags (e.g. `2025-016-04` — a typo), which crashed `datetime.strptime`.
  2. Newer posts switched to **client-side markdown rendering**: the HTML `<main>` is empty and loads `markdown-posts/*.md` via `marked.js` + KaTeX. So the old `<main>`-parsing extracted nothing.
  3. It was run once by hand and never re-run.
- Date inconsistencies exist across `blog.html` (the index), the posts' meta tags, and markdown front matter. `blog.html` lists some posts as `2026-04-22` while their meta tags say `2025-06-16`.

## Decisions made

1. **pandoc** converts markdown → HTML (user's idea; pandoc 3.2 is installed via Homebrew). It handles tables/code/footnotes and, unlike Python's `markdown`, could also read HTML if needed later.
2. **Full post content in the feed**, not excerpts. Full HTML goes into `<content:encoded>` only — a single copy (the user rejected storing duplicates in both `<description>` and `<content:encoded>`). `<title>`, `<link>`, `<guid>`, and `<pubDate>` remain for readers that don't render `content:encoded`.
3. **`blog.html` is the source of truth for URL resolution** (the post URL is derived by matching a markdown file's basename against the `<a href>` entries in `blog.html`).
4. **YAML front matter is the source of truth for `title` and `date`** in each markdown post:
   ```markdown
   ---
   title: My post title
   date: YYYY-MM-DD
   ---
   ```
   Falls back to (title: `#` heading → `blog.html` anchor → filename; date: `blog.html` entry).
5. **Limit to newest N posts** via `MAX_ITEMS` (default `20`, `0` = unlimited) so the feed doesn't grow unboundedly while carrying full content.
6. **GitHub Action auto-regenerates** the feed on push, so it can never go stale again.
7. **Markdown template** (`_template.md`, underscore-prefixed so the generator skips it).

## Files created / changed

### `make_feed.sh` (new, repo root, executable)
Bash script. Key points:
- Portable across macOS and Linux (the `date` RFC-822 conversion uses a `case "$(uname -s)"` branch — macOS `date -j -f`, GNU `date -d`).
- Config at top: `BASE_URL`, `FEED_TITLE`/`FEED_LINK`/`FEED_DESC`, `BLOG_INDEX=blog.html`, `POSTS_DIR=blog/markdown-posts`, `OUT=blog/feed.xml`, `MAX_ITEMS` (env-overridable).
- **Skips** any `_`-prefixed `.md` (templates/drafts) and any file with no date.
- For each post: reads front matter (`title`/`date`), resolves URL from `blog.html`, runs `pandoc "$md" -f markdown -t html5`, escapes title/url, writes one `<item>` per file into a temp dir keyed `DATE_SLUG.xml`.
- Assembles channel header + items sorted newest-first (`ls -r`), writes `blog/feed.xml`.
- `content:encoded` uses CDATA; `]]>` is escaped to avoid breaking CDATA.

### `.github/workflows/update-feed.yml` (new)
- Trigger: push to `main` with `paths` filter on `blog/markdown-posts/**`, `blog/*.html`, `blog.html`, `make_feed.sh`; plus `workflow_dispatch`.
- `permissions: contents: write`.
- Steps: `actions/checkout@v4` → `apt-get install pandoc` → `bash make_feed.sh` → commit+push `blog/feed.xml` only if changed (commit message "Regenerate RSS feed [skip ci]").
- `paths` excludes `feed.xml`, so the regenerated commit won't retrigger the workflow (no loop).

### `blog/markdown-posts/_template.md` (new)
Copyable template with `title`/`date` front matter placeholders, placeholder sections, and a reminder to use absolute URLs for images/links (so they resolve in RSS readers).

### `blog/markdown-posts/drug-like-lacan-csvbase.md` (edited)
Added front matter block:
```markdown
---
title: Drug-like LACAN mols on csvbase
date: 2026-08-18
---
```

### `blog/feed.xml` (regenerated)
Now contains the single markdown post (#34) with full HTML content. Valid XML (verified with `xml.etree.ElementTree`).

## How to add a new post (going forward)

1. `cp blog/markdown-posts/_template.md blog/markdown-posts/my-post.md`
2. Fill in `title` and `date` (real `YYYY-MM-DD` — this is what marks it "new" in readers).
3. Write the post (prefer absolute URLs for images/links).
4. Add an entry to `blog.html` (date + title + link to the HTML page) and create the post's HTML wrapper if using the markdown-rendering template.
5. Commit and push. The GitHub Action regenerates `feed.xml` automatically.

Or run locally: `bash make_feed.sh` (optionally `MAX_ITEMS=10 bash make_feed.sh`).

## Config knobs to remember

- `MAX_ITEMS` (default `20`) — how many recent posts the feed keeps.
- `FEED_TITLE` / `FEED_DESC` — feed metadata.
- RSS math caveat (not yet implemented): KaTeX math won't render in RSS readers; if it ever matters, pre-render or use `pandoc --mathml`.

## Pre-existing oddities (not addressed, just noted)

- `git status` showed `blog/X_duckdb_logreg.html` and `blog/_parallel_zinc.html` as already-deleted in the working tree (we did not delete them).
- Underscore/x-prefixed files (`_parallel_zinc.html`, `X_duckdb_logreg.html`) and `blog/graveyard/` appear to be the user's convention for drafts/retired posts.

## Status

Working and confirmed end-to-end: a new post appeared in the user's NetNewsWire app after the feed was committed and pushed.