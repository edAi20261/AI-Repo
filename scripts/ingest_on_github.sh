#!/usr/bin/env bash
# Ingest a shelf of books on GitHub's machines. This computer only sends the
# request — nothing is downloaded, rendered or embedded here.
#
#   scripts/ingest_on_github.sh books.txt
#
# books.txt is one link per line: a Google Drive link shared as "Anyone with the
# link", an https URL or an s3:// URI. Blank lines and lines starting with # are
# skipped. Everything else about a book — its ids, title and language — is
# worked out from the book itself.
#
# Needs the GitHub CLI (https://cli.github.com), logged in once with
# `gh auth login` as someone who can run workflows in this repository.
set -euo pipefail

list="${1:-}"
if [ -z "$list" ] || [ ! -f "$list" ]; then
  echo "usage: $0 <books.txt>" >&2
  exit 2
fi
command -v gh >/dev/null 2>&1 || { echo "the GitHub CLI (gh) is not installed" >&2; exit 2; }
gh auth status >/dev/null 2>&1 || { echo "run 'gh auth login' first" >&2; exit 2; }

started=0
while IFS= read -r link || [ -n "${link:-}" ]; do
  link="$(printf '%s' "$link" | tr -d '\r' | xargs)"
  case "$link" in ""|\#*) continue;; esac
  gh workflow run ingest-book.yml --ref "${INGEST_REF:-main}" -f source="$link"
  echo "started  $link"
  started=$((started + 1))
done < "$list"

echo
echo "$started book(s) started. Watch them in the repository's Actions tab, under"
echo "'Ingest a book', or with: gh run list --workflow ingest-book.yml"
