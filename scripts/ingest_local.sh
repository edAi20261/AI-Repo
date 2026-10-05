#!/usr/bin/env bash
# Ingest a shelf of books on one machine, using the production image.
#
#   scripts/ingest_local.sh books.tsv [extra ingestion arguments]
#
# books.tsv is one book per line, tab separated, comments and blanks ignored:
#
#   /books/unit1.pdf<TAB>LECTURE_ID<TAB>SUBJECT_ID<TAB>Book title
#
# The same manifest ingest_batch.sh takes, except the source is a file on this
# machine rather than an S3 URI — the point of running here is that the PDF is
# already here.
#
# Why the same container the cloud runs rather than a local Python: the image
# carries the embedding model and a pinned set of libraries, and the one thing
# nobody can afford is a book indexed differently depending on whose machine
# read it. Docker is also what makes this work the same on Windows, which is
# where it is expected to run.
#
# One book at a time by default. A book holds its rendered pages in memory
# while it reads them — about 1.2 GB for four hundred pages — on top of the
# embedding model, and a laptop with 16 GB does not have room for three of
# those plus an operating system.
set -euo pipefail

IMAGE="${INGEST_IMAGE:-edu-ai-ingest:local}"
ENV_FILE="${INGEST_ENV_FILE:-.env.ingest}"
AT_ONCE="${INGEST_AT_ONCE:-1}"
BOOKS_DIR="${INGEST_BOOKS_DIR:-}"
LOG_DIR="${INGEST_LOG_DIR:-ingest-logs}"

manifest="${1:-}"
if [ -z "$manifest" ] || [ ! -f "$manifest" ]; then
  echo "usage: $0 <books.tsv> [extra ingestion arguments]" >&2
  exit 2
fi
shift
extra=()
[ "$#" -gt 0 ] && extra=("$@")

for needed in docker; do
  command -v "$needed" >/dev/null 2>&1 || { echo "$needed is not installed" >&2; exit 2; }
done
if [ ! -f "$ENV_FILE" ]; then
  echo "no $ENV_FILE — copy .env.ingest.example and fill it in" >&2
  exit 2
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "no $IMAGE image yet. Build it once with:" >&2
  echo "  docker build -f Dockerfile.ingest -t $IMAGE ." >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
started=$(date -u '+%Y%m%d-%H%M%S')
summary="$LOG_DIR/run-$started.log"
: > "$summary"

# A credential this run cannot use is better found now than on page three
# hundred of the first book.
echo "checking the credentials before reading anything…"
if ! docker run --rm --env-file "$ENV_FILE" "$IMAGE" -m scripts.preflight_ingest 2>&1 | tee -a "$summary"; then
  echo "preflight failed — nothing was ingested" >&2
  exit 1
fi

running=0
declare -a failed=()

ingest_one() {
  local source="$1" lecture="$2" subject="$3" title="$4"
  local name mount log
  # The whole id, not a prefix of it. Books from one shelf are given ids that
  # differ in the last characters, so a short name put two books in one log and
  # the first one's was overwritten without a word.
  name="ingest-$lecture"
  log="$LOG_DIR/$lecture.log"

  # The book is mounted read only at a fixed path, so the manifest can hold a
  # Windows path, a WSL path or a relative one and the container sees one shape.
  local dir base
  dir="$(cd "$(dirname "$source")" && pwd)"
  base="$(basename "$source")"
  mount="$dir"

  echo "started  $title  -> $log" | tee -a "$summary"
  if docker run --rm --name "$name" \
      --env-file "$ENV_FILE" \
      --volume "$mount:/books:ro" \
      --memory "${INGEST_MEMORY:-10g}" \
      "$IMAGE" -m ingestion "/books/$base" \
        --lecture-id "$lecture" --subject-id "$subject" --lesson-title "$title" \
        ${extra[@]+"${extra[@]}"} > "$log" 2>&1; then
    echo "finished $title" | tee -a "$summary"
  else
    echo "FAILED   $title  (see $log)" | tee -a "$summary"
    failed+=("$title")
  fi
}

while IFS=$'\t' read -r source lecture subject title || [ -n "${source:-}" ]; do
  case "${source:-}" in ""|\#*) continue;; esac
  [ -n "$BOOKS_DIR" ] && [ "${source#/}" = "$source" ] && source="$BOOKS_DIR/$source"
  if [ ! -f "$source" ]; then
    echo "FAILED   $title  (no such file: $source)" | tee -a "$summary"
    failed+=("$title")
    continue
  fi

  if [ "$AT_ONCE" -le 1 ]; then
    ingest_one "$source" "$lecture" "$subject" "$title"
  else
    ingest_one "$source" "$lecture" "$subject" "$title" &
    running=$((running + 1))
    [ "$running" -ge "$AT_ONCE" ] && { wait -n 2>/dev/null || wait; running=$((running - 1)); }
  fi
done < "$manifest"
wait

echo | tee -a "$summary"
if [ "${#failed[@]}" -eq 0 ]; then
  echo "all books finished. Now check they were indexed:" | tee -a "$summary"
  echo "  docker run --rm --env-file $ENV_FILE -v \"\$PWD:/w\" $IMAGE \\" | tee -a "$summary"
  echo "    -m scripts.ingest_status /w/$(basename "$manifest")" | tee -a "$summary"
else
  echo "${#failed[@]} book(s) failed:" | tee -a "$summary"
  printf '  %s\n' "${failed[@]}" | tee -a "$summary"
  echo "Re-running the same manifest resumes them: a page already read is not read again." \
    | tee -a "$summary"
  exit 1
fi
