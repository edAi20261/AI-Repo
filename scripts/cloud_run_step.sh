#!/usr/bin/env bash
# One step of a book's ingestion, as a Cloud Run job task. The same commands the
# ingest-book workflow runs, so a book is ingested identically on either.
#
#   SOURCE        the book: a Drive link, an https URL or an s3:// URI
#   SUBJECT_ID    the subject the book's lessons are created under
#   RUN_DIR       this run's folder on the work bucket, mounted at /work
#
# A task learns its share from Cloud Run, which numbers the tasks of an
# execution: CLOUD_RUN_TASK_INDEX of CLOUD_RUN_TASK_COUNT.
set -euo pipefail

step="${1:?usage: cloud_run_step.sh read|prepare|embed|finish}"
shard="${CLOUD_RUN_TASK_INDEX:-0}"
of="${CLOUD_RUN_TASK_COUNT:-1}"
run_dir="${RUN_DIR:?RUN_DIR names the folder of this run on the work bucket}"
book=/tmp/book.pdf

fetch() {
  python -m scripts.fetch_book "${SOURCE:?SOURCE names the book}" "$book"
  # shellcheck source=scripts/ci_lesson_args.sh
  source scripts/ci_lesson_args.sh
}

case "$step" in
  read)
    python -m scripts.preflight_ingest --skip embedder
    fetch
    python -m ingestion.distributed read "$book" "${LESSON_ARGS[@]}" \
      --shard "$shard" --of "$of" --workers "${READ_WORKERS:-8}"
    ;;
  prepare)
    fetch
    python -m ingestion.distributed prepare "$book" "${LESSON_ARGS[@]}" \
      --out "$run_dir/state/state.pkl"
    ;;
  embed)
    python -m ingestion.distributed embed --state "$run_dir/state/state.pkl" \
      --shard "$shard" --of "$of" --out "$run_dir/vectors/vectors-$shard.pkl"
    ;;
  finish)
    python -m ingestion.distributed finish --state "$run_dir/state/state.pkl" \
      --vectors "$run_dir/vectors"
    ;;
  *)
    echo "unknown step: $step" >&2
    exit 2
    ;;
esac
