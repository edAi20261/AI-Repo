# Sourced by the ingest-book workflow: the lesson's arguments, built from the
# workflow's inputs as an array so a title with spaces or quotes stays one word.
# Inputs arrive as environment variables, never pasted into the script — a
# lesson title is text a person typed.
LESSON_ARGS=(--lecture-id "$LECTURE_ID" --subject-id "$SUBJECT_ID")
[ -n "${LESSON_TITLE:-}" ] && LESSON_ARGS+=(--lesson-title "$LESSON_TITLE")
case "${LANGUAGE:-}" in ""|"ask the backend") ;; *) LESSON_ARGS+=(--language "$LANGUAGE");; esac
[ -n "${GRADE:-}" ] && LESSON_ARGS+=(--grade "$GRADE")
[ "${SUBJECT_IS_MATH:-false}" = "true" ] && LESSON_ARGS+=(--subject-is-math)
true
