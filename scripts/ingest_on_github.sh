#!/usr/bin/env bash
# Ingest a shelf of books on GitHub's machines. This computer only sends the
# request — nothing is downloaded, rendered or embedded here.
#
#   scripts/ingest_on_github.sh books.txt
#
# books.txt is one book per line: its link (a Google Drive link shared as
# "Anyone with the link", an https URL or an s3:// URI), the id of the subject
# it belongs to, and what it is: ministry (the app shows its lessons) or
# external (background for the subject's ministry lessons, never shown).
# Blank lines and lines starting with # are skipped.
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
while IFS= read -r line || [ -n "${line:-}" ]; do
  line="$(printf '%s' "$line" | tr -d '\r' | xargs)"
  case "$line" in ""|\#*) continue;; esac
  read -r link subject kind _ <<<"$line"
  case "${kind:-}" in
    ministry|external) ;;
    *) echo "skipped  $link (needs: link subject_id ministry|external)" >&2; continue ;;
  esac
  gh workflow run ingest-book.yml --ref "${INGEST_REF:-main}" -f source="$link" \
    -f subject_id="$subject" -f kind="$kind"
  echo "started  $link -> subject $subject ($kind)"
  started=$((started + 1))
done < "$list"

echo
echo "$started book(s) started. Watch them in the repository's Actions tab, under"
echo "'Ingest a book', or with: gh run list --workflow ingest-book.yml"
