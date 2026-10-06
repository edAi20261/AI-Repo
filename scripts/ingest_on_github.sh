#!/usr/bin/env bash
# Ingest a shelf of books on GitHub's machines. This computer only sends the
# request — nothing is downloaded, rendered or embedded here.
#
#   scripts/ingest_on_github.sh books.tsv
#
# books.tsv is one book per line, tab separated, comments and blanks ignored:
#
#   LINK<TAB>language[<TAB>Subject name[<TAB>Book title]]
#
# LINK is a Google Drive link shared as "Anyone with the link", an https URL or
# an s3:// URI. language is ar, en, fr or de. Subject name groups books (e.g.
# English); the title defaults to the file's name. The lecture and subject IDs
# are worked out from the book itself, so the same book always replaces itself.
#
# Needs the GitHub CLI (https://cli.github.com), logged in once with
# `gh auth login` as someone who can run workflows in this repository.
#
#   INGEST_REF=main            the branch whose code reads the books
#   INGEST_READ_MACHINES=6     machines reading pages, per book
#   INGEST_EMBED_MACHINES=8    machines embedding chunks, per book
set -euo pipefail

manifest="${1:-}"
if [ -z "$manifest" ] || [ ! -f "$manifest" ]; then
  echo "usage: $0 <books.tsv>" >&2
  exit 2
fi
command -v gh >/dev/null 2>&1 || { echo "the GitHub CLI (gh) is not installed" >&2; exit 2; }
gh auth status >/dev/null 2>&1 || { echo "run 'gh auth login' first" >&2; exit 2; }

ref="${INGEST_REF:-main}"
started=0
while IFS=$'\t' read -r source language subject title || [ -n "${source:-}" ]; do
  case "${source:-}" in ""|\#*) continue;; esac
  case "${language:-}" in ar|en|fr|de) ;;
    *) echo "skipped: give the book's language (ar, en, fr or de) after its link: $source" >&2
       continue;;
  esac
  gh workflow run ingest-book.yml --ref "$ref" \
    -f source="$source" \
    -f language="$language" \
    -f subject_name="${subject:-}" \
    -f lesson_title="${title:-}" \
    -f read_machines="${INGEST_READ_MACHINES:-6}" \
    -f embed_machines="${INGEST_EMBED_MACHINES:-8}"
  echo "started  ${title:-$source}"
  started=$((started + 1))
done < "$manifest"

echo
echo "$started book(s) started. Watch them with:"
echo "  gh run list --workflow ingest-book.yml"
echo "or in the repository's Actions tab, under 'Ingest a book'."
