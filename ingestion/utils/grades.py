"""Normalise a grade label to the value ``academics.grade_level`` accepts.

The platform's API serialises the enum by name (``GRADE_7``) while the database
column stores its value (``Grade 7``), and fixtures use short forms like ``g8``.
Everything written to ``ai.*`` and to the vector payload goes through here, so a
grade filter matches whichever spelling the source used.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

MIN_GRADE, MAX_GRADE = 1, 12
GRADE_NUMBER = re.compile(r"^(?:grade|g|year|yr)?[\s_-]*(\d{1,2})$", re.IGNORECASE)


def to_platform_grade(value: Optional[str]) -> Optional[str]:
    """``GRADE_7`` | ``g7`` | ``7`` | ``Grade 7`` -> ``Grade 7``; unknown -> ``None``."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    match = GRADE_NUMBER.match(raw)
    number = int(match.group(1)) if match else None
    if number is not None and MIN_GRADE <= number <= MAX_GRADE:
        return f"Grade {number}"

    logger.warning("unrecognised grade label, storing none", extra={"grade": raw})
    return None
