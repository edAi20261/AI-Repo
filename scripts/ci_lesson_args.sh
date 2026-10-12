# Sourced by the ingest-book workflow: the lesson's arguments, built from the
# workflow's inputs as an array so a title with spaces or quotes stays one word.
# Inputs arrive as environment variables, never pasted into the script — a
# lesson title is text a person typed.
# Anything left empty is worked out from the book itself, identically on every
# machine (ingestion/lesson.py, fill_in_identity).
LESSON_ARGS=()
[ -n "${LECTURE_ID:-}" ] && LESSON_ARGS+=(--lecture-id "$LECTURE_ID")
[ -n "${SUBJECT_ID:-}" ] && LESSON_ARGS+=(--subject-id "$SUBJECT_ID")
[ -n "${SUBJECT_NAME:-}" ] && LESSON_ARGS+=(--subject-name "$SUBJECT_NAME")
[ -n "${LESSON_TITLE:-}" ] && LESSON_ARGS+=(--lesson-title "$LESSON_TITLE")
case "${LANGUAGE:-}" in ""|"ask the backend") ;; *) LESSON_ARGS+=(--language "$LANGUAGE");; esac
[ -n "${GRADE:-}" ] && LESSON_ARGS+=(--grade "$GRADE")
[ "${SUBJECT_IS_MATH:-false}" = "true" ] && LESSON_ARGS+=(--subject-is-math)
case "${BOOK_KIND:-ministry}" in
  ministry) ;;
  external) LESSON_ARGS+=(--namespace external) ;;
  *) echo "BOOK_KIND is ministry or external, not ${BOOK_KIND}" >&2; return 2 ;;
esac
true
