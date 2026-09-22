"""Failures a turn reports instead of papering over."""

from __future__ import annotations

from shared.errors import ServiceUnavailableError


class TutorUnavailable(ServiceUnavailableError):
    """A model the turn depends on did not answer.

    Raised rather than answered with a fixed apology: a sentence written in the
    service is in some language, and the client already shows its own message
    for a failed request, in the language its interface is in.
    """

    def __init__(self) -> None:
        super().__init__("The tutor could not answer right now. Please try again.")
