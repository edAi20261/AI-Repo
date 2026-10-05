"""Remembering what a model saw on a page.

One vision call per page is the most expensive thing this pipeline does, and
chunking is the thing most often changed. Coupling them means every adjustment
to how blocks are grouped re-pays for reading every page, which is enough to
stop anyone adjusting it.

So the parse is stored as the model returned it. Re-chunking reads from here and
costs nothing. A page goes back to a model only when the model changes or the
prompt does — and the prompt's version is a digest of its own text, so changing
a word invalidates the cache without anyone remembering to say so.
"""

from __future__ import annotations

import logging
from typing import Optional

from psycopg.types.json import Json
from pydantic import BaseModel

from shared.config import Config
from shared.postgres import connect

from .blocks import ParsedPage

logger = logging.getLogger(__name__)

# Batch work: a cache read must not be held to a student turn's budget, and a
# whole-document write is meant to take as long as it takes.
CONNECT_TIMEOUT = 15
NO_STATEMENT_TIMEOUT = 0

_SELECT = """
SELECT raw_output FROM ai.page_transcriptions
WHERE lecture_id = %s::uuid AND page = %s AND model = %s AND prompt_version = %s
"""

_UPSERT = """
INSERT INTO ai.page_transcriptions
    (lecture_id, page, model, prompt_version, raw_output, image_uri, dpi)
VALUES (%s::uuid, %s, %s, %s, %s, %s, %s)
ON CONFLICT (lecture_id, page, model, prompt_version)
DO UPDATE SET raw_output = EXCLUDED.raw_output,
              image_uri  = EXCLUDED.image_uri,
              dpi        = EXCLUDED.dpi,
              created_at = CURRENT_TIMESTAMP
"""


class PageCache:
    """Parsed pages, keyed by what produced them.

    A cache that cannot be reached is not an error worth failing an ingest over,
    but it is never silent: the run costs full price and the reason is logged
    once rather than per page.
    """

    def __init__(self, config: Config, *, model: str, prompt_version: str,
                 shape: type[BaseModel] = ParsedPage) -> None:
        # What a stored page is read back as. A page read for its layout and a
        # page read for its words are different objects that cost the same model
        # call, so both belong in this table — under their own prompt, and read
        # back into their own shape. Storing one and validating it as the other
        # serves a run the parse it did not ask for.
        self._config = config
        self._model = model
        self._prompt_version = prompt_version
        self._shape = shape
        self._available = True
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._available

    def get(self, lecture_id: str, page: int) -> Optional[BaseModel]:
        """The stored parse for this page, or ``None`` if it has to be read again."""
        if not self._available:
            return None
        rows = self._run(_SELECT, (lecture_id, page, self._model,
                                   self._prompt_version), fetch=True)
        if not self._available:
            # An outage is not a miss. Counting it as one reports a cache that
            # is working and simply cold, which is the opposite of what happened.
            return None
        if not rows:
            self.misses += 1
            return None
        if not self._is_our_shape(rows[0][0]):
            # Every field of a transcription has a default, so a layout parse
            # validates into one with no words at all — and a page indexed as
            # empty looks exactly like a page that was genuinely blank. The two
            # kinds of page live in one table under their own prompt, which is
            # what keeps them apart; this is the check that does not depend on
            # anyone remembering that.
            logger.warning("a cached page of another kind was ignored",
                           extra={"lecture_id": lecture_id, "page": page,
                                  "shape": self._shape.__name__})
            self.misses += 1
            return None
        try:
            parsed = self._shape.model_validate(rows[0][0])
        except Exception:
            # A stored parse that no longer fits the schema is stale, not fatal:
            # the shape changed under it. Read the page again and overwrite.
            logger.warning("cached page no longer fits the schema; re-reading",
                           extra={"lecture_id": lecture_id, "page": page})
            self.misses += 1
            return None
        self.hits += 1
        return parsed

    def put(
        self,
        lecture_id: str,
        page: int,
        parsed: BaseModel,
        *,
        image_uri: Optional[str] = None,
        dpi: Optional[int] = None,
    ) -> None:
        if not self._available:
            return
        self._run(_UPSERT, (lecture_id, page, self._model, self._prompt_version,
                            Json(parsed.model_dump(mode="json")), image_uri, dpi))

    def _is_our_shape(self, stored: object) -> bool:
        """Whether this row was written by something storing ``self._shape``.

        Exact rather than a guess, because what goes in is a ``model_dump`` and
        a model emits only its own declared fields — parse liberally, emit
        strictly, so a row holding a field this shape never declares was
        written by a different one.
        """
        if not isinstance(stored, dict):
            return False
        return bool(stored) and set(stored) <= set(self._shape.model_fields)

    def _run(self, sql: str, params: tuple, *, fetch: bool = False):
        try:
            with connect(self._config, connect_timeout=CONNECT_TIMEOUT,
                         statement_timeout_ms=NO_STATEMENT_TIMEOUT) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall() if fetch else None
                conn.commit()
                return rows
        except Exception as error:      # noqa: BLE001 — never fail an ingest over a cache
            self._available = False
            logger.error(
                "page cache unavailable; this run pays full price for every page. "
                "Apply ai.page_transcriptions from scripts/dev_ai_schema.sql.",
                exc_info=error)
            return None

    def report(self) -> dict:
        """What the cache did, for the run's stats."""
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "cache_enabled": self._available}
