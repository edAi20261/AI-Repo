"""Language policy: precedence, refusal, catalogue coverage and verification.

The named regressions are the two failures that prompted this layer — a student
asked "explain column subtraction" and was answered in French.
"""

from __future__ import annotations

import pytest

from shared.language import (
    LanguageDecision,
    LanguagePolicy,
    classify,
    content_directive,
    explicit_request,
    language_directive,
    looks_arabizi,
    translate,
    verify_answer,
)
from shared.language.catalog import STRINGS
from shared.language.verify import prose_of

SUPPORTED = ("ar", "en", "fr", "de")


@pytest.fixture
def policy() -> LanguagePolicy:
    return LanguagePolicy(supported=SUPPORTED, default="en")


# --- the reported bugs ------------------------------------------------------

def test_short_english_question_is_not_french(policy):
    """The reported bug: 24 letters of English scored 0.57 for French."""
    assert policy.resolve("explain column subtraction").code == "en"


def test_single_word_greeting_is_not_finnish(policy):
    assert policy.resolve("hello").code == "en"


# --- precedence -------------------------------------------------------------

def test_explicit_request_beats_everything(policy):
    decision = policy.resolve(
        "explain photosynthesis, answer in French",
        api_language="ar", profile_language="de", previous_language="en",
    )
    assert (decision.code, decision.source) == ("fr", "explicit")


def test_api_language_is_honoured(policy):
    """The parameter used to be written to state and overwritten on the next line."""
    decision = policy.resolve("اشرح لي عملية البناء الضوئي", api_language="fr")
    assert (decision.code, decision.source) == ("fr", "api")


def test_script_beats_detection(policy):
    decision = policy.resolve("اشرح لي الـ photosynthesis")
    assert (decision.code, decision.source) == ("ar", "script")


def test_planner_hint_only_fills_what_detection_declined(policy):
    confident = policy.resolve("Erklären Sie mir bitte die Photosynthese", planner_hint="ar")
    assert confident.code == "de"

    declined = policy.resolve("ok", planner_hint="ar")
    assert (declined.code, declined.source) == ("ar", "planner")


def test_profile_then_previous_then_default(policy):
    assert policy.resolve("ok", profile_language="fr").source == "profile"
    assert policy.resolve("ok", previous_language="de").source == "previous"
    assert policy.resolve("ok").source == "default"


def test_unsupported_values_are_ignored(policy):
    decision = policy.resolve("explain column subtraction", api_language="es")
    assert decision.code == "en"


# --- refusal ----------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "momken tefhemni el dars da?",
    "ezay a7seb el mesaha?",
    "3ayez a3raf el drs dah",
])
def test_arabizi_is_refused_not_guessed(policy, text):
    """Every detector measured returned a European language for these."""
    decision = policy.resolve(text, profile_language="ar")
    assert decision.refused is True
    assert decision.script == "arabizi"
    assert decision.code == "ar"


def test_arabizi_can_be_routed_to_arabic_by_config():
    policy = LanguagePolicy(supported=SUPPORTED, default="en", arabizi_hint="ar")
    assert policy.resolve("ezay a7seb el mesaha?").code == "ar"


def test_a_refused_decision_is_not_verified(policy):
    decision = policy.resolve("ok")
    assert decision.refused is True
    assert decision.verifiable is False


def test_text_without_letters_keeps_continuity(policy):
    assert policy.resolve("2 + 2 = ?", previous_language="ar").code == "ar"


# --- the explicit-request matcher -------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("answer in French: what is subtraction", "fr"),
    ("reply in Arabic please", "ar"),
    ("اشرح لي الدرس بالعربي", "ar"),
    ("réponds en arabe s'il te plaît", "ar"),
    ("bitte antworte auf Deutsch", "de"),
    ("translate into English", "en"),
])
def test_requests_are_recognised(text, expected):
    assert explicit_request(text)[0] == expected


@pytest.mark.parametrize("text", [
    "what happened in the French Revolution?",
    "I need help with my English literature homework",
    "we studied the German empire in class",
])
def test_a_topic_is_not_a_request(text):
    """A language name is a subject here, not an instruction."""
    assert explicit_request(text) is None


# --- script -----------------------------------------------------------------

def test_script_classification():
    assert classify("ما هي الجملة الاسمية؟") == "arabic"
    assert classify("what is a noun") == "latin"
    assert classify("a7seb el mesaha") == "arabizi"
    assert classify("") == "none"


def test_arabizi_needs_evidence():
    assert looks_arabizi("a7seb") is True
    assert looks_arabizi("hello there") is False


# --- catalogue --------------------------------------------------------------

def test_every_string_exists_in_every_supported_language():
    missing = {
        key: sorted(set(SUPPORTED) - set(entries))
        for key, entries in STRINGS.items()
        if set(SUPPORTED) - set(entries)
    }
    assert not missing, f"untranslated: {missing}"


def test_no_translation_is_a_copy_of_the_english():
    """A copy-pasted English string reads as a translation and is never noticed."""
    copied = [
        key for key, entries in STRINGS.items()
        if entries.get("ar") == entries.get("en") and not _is_language_neutral(entries["en"])
    ]
    assert not copied, f"Arabic entries identical to English: {copied}"


def _is_language_neutral(text: str) -> bool:
    return not any(char.isalpha() for char in text) or text.strip() in {"PDF"}


def test_a_missing_key_is_reported_not_silently_english(caplog):
    with caplog.at_level("ERROR", logger="shared.language.catalog"):
        assert translate("nope.not.a.key", "ar") == "nope.not.a.key"
    assert "i18n missing key" in caplog.text


def test_placeholders_are_filled():
    assert "5" in translate("compose.assessment_line", "en", n=5)
    assert "5" in translate("compose.assessment_line", "ar", n=5)


# --- the directive ----------------------------------------------------------

def test_directive_names_the_language_in_both_scripts():
    directive = language_directive("ar")
    assert "Arabic" in directive and "العربية" in directive
    assert "source material" in directive
    assert "technical terms" in directive


def test_content_directive_is_for_material_not_conversation():
    assert "student" not in content_directive("en")


# --- verification -----------------------------------------------------------

ARABIC_ANSWER = (
    "الطرح بالأعمدة يعني أن نكتب العدد الأكبر في الأعلى والعدد الأصغر في الأسفل، "
    "ثم نطرح الآحاد أولًا ثم العشرات."
)
FRENCH_ANSWER = (
    "La soustraction en colonnes consiste à écrire le plus grand nombre en haut et "
    "le plus petit en bas, puis à soustraire les unités avant les dizaines."
)


def _decision(code: str, refused: bool = False) -> LanguageDecision:
    return LanguageDecision(code=code, source="script", confidence=1.0,
                            script="arabic", refused=refused)


def test_mismatch_is_detected():
    check = verify_answer(FRENCH_ANSWER, _decision("ar"), languages=SUPPORTED)
    assert check.needs_repair is True
    assert check.detected == "fr"


def test_matching_answer_passes():
    assert verify_answer(ARABIC_ANSWER, _decision("ar"), languages=SUPPORTED).needs_repair is False


def test_a_guessed_language_is_never_repaired():
    check = verify_answer(FRENCH_ANSWER, _decision("ar", refused=True), languages=SUPPORTED)
    assert check.needs_repair is False


def test_short_answers_are_not_judged():
    assert verify_answer("نعم.", _decision("ar"), languages=SUPPORTED).needs_repair is False


def test_prose_strips_what_carries_no_language():
    stripped = prose_of("The answer is $x^2 + 1$ [1] see `code` https://example.com")
    assert "$" not in stripped and "[1]" not in stripped and "http" not in stripped
