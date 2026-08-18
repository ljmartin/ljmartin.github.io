#!/usr/bin/env bash
set -euo pipefail

# Generates blog/feed.xml from the markdown posts under blog/markdown-posts/.
# Runs locally (macOS) and in GitHub Actions (Linux); date handling is
# portable across both. Full post content is emitted in each item.

BASE_URL="https://ljmartin.github.io"
FEED_TITLE="LJM CompMedChem"
FEED_LINK="https://ljmartin.github.io/"
FEED_DESC="Compchem side projects and code snippets"
BLOG_INDEX="blog.html"
POSTS_DIR="blog/markdown-posts"
OUT="blog/feed.xml"

cd "$(dirname "$0")"

# --- XML-escape a string for use as element text ---------------------------
escape() {
  printf '%s' "$1" \
    | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

# --- Convert YYYY-MM-DD to RFC-822 (RSS <pubDate>) -------------------------
to_rfc822() {
  case "$(uname -s)" in
    Darwin) date -j -f "%Y-%m-%d" "$1" "+%a, %d %b %Y 00:00:00 +0000" ;;
    *)      date -d "$1" "+%a, %d %b %Y 00:00:00 +0000" ;;
  esac
}

# --- Read a field from pandoc-style YAML front matter ----------------------
frontmatter() {
  awk -v k="$1" '
    $0 ~ "^"k":" {
      sub("^"k":[[:space:]]*", "")
      sub(/^"/, ""); sub(/"$/, "")
      print; exit
    }' "$2"
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# --- Build one <item> per post ----------------------------------------------
for md in "$POSTS_DIR"/*.md; do
  [ -e "$md" ] || continue
  base="$(basename "$md" .md)"

  # title: front matter > first `# ` heading > blog.html anchor > filename
  title="$(frontmatter title "$md")"
  [ -z "$title" ] && title="$(awk '/^# / { sub(/^# /, ""); print; exit }' "$md")"

  # date: front matter > blog.html entry
  pubdate="$(frontmatter date "$md")"

  # link/url: match the post's basename in the blog index
  url=""; idx_title=""; idx_date=""
  line="$(grep -F "$base" "$BLOG_INDEX" | head -1 || true)"
  if [ -n "$line" ]; then
    href="$(printf '%s' "$line" | grep -oE 'href="blog/[^"]+\.html"' | head -1 | sed -E 's/^href="//; s/"$//')"
    [ -n "$href" ] && url="${BASE_URL}/${href}"
    idx_title="$(printf '%s' "$line" | sed -E 's/.*<a href="[^"]+">([^<]*)<\/a>.*/\1/')"
    idx_date="$(printf '%s' "$line" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1)"
  fi
  [ -z "$url" ] && url="${BASE_URL}/blog/${base}.html"

  [ -z "$title" ] && title="$idx_title"
  [ -z "$title" ] && title="$base"
  [ -z "$pubdate" ] && pubdate="$idx_date"
  if [ -z "$pubdate" ]; then
    echo "WARNING: no date for $md; skipping" >&2
    continue
  fi

  rfc="$(to_rfc822 "$pubdate")"

  # full content: markdown -> HTML fragment via pandoc
  content="$(pandoc "$md" -f markdown -t html5 | sed -e 's/]]>/]]\&gt;/g')"

  esc_title="$(escape "$title")"
  esc_url="$(escape "$url")"

  {
    printf '   <item>\n'
    printf '     <title>%s</title>\n' "$esc_title"
    printf '     <link>%s</link>\n' "$esc_url"
    printf '     <guid isPermaLink="true">%s</guid>\n' "$esc_url"
    printf '     <pubDate>%s</pubDate>\n' "$rfc"
    printf '     <description><![CDATA[%s]]></description>\n' "$content"
    printf '     <content:encoded><![CDATA[%s]]></content:encoded>\n' "$content"
    printf '   </item>\n'
  } > "${tmp}/${pubdate}_${base}.xml"
done

# --- Assemble the feed, newest first ----------------------------------------
{
  printf '<?xml version="1.0" encoding="UTF-8"?>\n'
  printf '<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:content="http://purl.org/rss/1.0/modules/content/">\n'
  printf '<channel>\n'
  printf '  <title>%s</title>\n' "$(escape "$FEED_TITLE")"
  printf '  <link>%s</link>\n' "$(escape "$FEED_LINK")"
  printf '  <description>%s</description>\n' "$(escape "$FEED_DESC")"
  printf '  <language>en</language>\n'
  printf '  <lastBuildDate>%s</lastBuildDate>\n' "$(to_rfc822 "$(date +%Y-%m-%d)")"
} > "$OUT"

# shellcheck disable=SC2045
for f in $(ls -r "$tmp"); do
  cat "${tmp}/$f" >> "$OUT"
done

printf '</channel>\n</rss>\n' >> "$OUT"

echo "Wrote $OUT ($(ls "$tmp" | wc -l | tr -d ' ') item(s))"