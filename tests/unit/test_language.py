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
    clear_request,
    content_directive,
    explicit_request,
    language_directive,
    language_instruction_only,
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
        api_language="ar", session_language="de", planner_hint="en",
    )
    assert (decision.code, decision.source) == ("fr", "explicit")


@pytest.mark.parametrize("text,expected", [
    ("explain subtraction, reply in Arabic", "ar"),
    ("explain photosynthesis. Reply in Arabic please", "ar"),
    ("Explain the water cycle, but in Arabic", "ar"),
    ("summarize this lesson, and write it in Arabic", "ar"),
    ("switch to Arabic", "ar"),
    ("continue in Arabic", "ar"),
    ("change to French", "fr"),
    ("please use Arabic", "ar"),
    ("اشرح لي الدرس بالانجليزي", "en"),
])
def test_a_request_at_the_end_of_a_message_is_obeyed(policy, text, expected):
    """The shape a student actually writes: the question in one language, the
    instruction tacked on."""
    assert policy.resolve(text).code == expected


@pytest.mark.parametrize("text", [
    "how do you say hello in Arabic?",
    "how to say thank you in French",
])
def test_asking_about_a_language_is_not_asking_for_it(policy, text):
    """The question is about Arabic; the answer belongs in the asker's language."""
    assert policy.resolve(text).code == "en"


def test_the_message_outranks_the_caller_s_parameter(policy):
    """The reported bug: the web app sent its interface language on every turn, so
    Arabic and French questions were both answered in English.

    Assistants people already use settle this the same way — Anthropic: "follow
    the language of the query unless explicitly requested otherwise", applied even
    over a stored "my native language is Spanish"."""
    decision = policy.resolve("اشرح لي عملية البناء الضوئي", api_language="en")
    assert (decision.code, decision.source) == ("ar", "script")


def test_the_parameter_decides_what_the_message_cannot(policy):
    """It is a default for an undecidable message, not an override of a clear one."""
    decision = policy.resolve("ok", api_language="fr")
    assert (decision.code, decision.source) == ("fr", "api")


def test_a_request_in_the_message_beats_the_parameter(policy):
    """An English question ending in "reply in Arabic" is answered in Arabic."""
    decision = policy.resolve("explain subtraction, reply in Arabic", api_language="en")
    assert (decision.code, decision.source) == ("ar", "explicit")


def test_script_beats_detection(policy):
    decision = policy.resolve("اشرح لي الـ photosynthesis")
    assert (decision.code, decision.source) == ("ar", "script")


def test_planner_hint_only_fills_what_detection_declined(policy):
    confident = policy.resolve("Erklären Sie mir bitte die Photosynthese", planner_hint="ar")
    assert confident.code == "de"

    declined = policy.resolve("ok", planner_hint="ar")
    assert (declined.code, declined.source) == ("ar", "planner")


def test_the_last_resort_is_the_default(policy):
    """There were two tiers between the planner hint and the default — a stored
    profile preference and the previous turn's language. Nothing ever supplied
    either, so both arrived None on every call the service ever made."""
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
    decision = policy.resolve(text)
    assert decision.refused is True
    assert decision.script == "arabizi"


def test_arabizi_can_be_routed_to_arabic_by_config():
    policy = LanguagePolicy(supported=SUPPORTED, default="en", arabizi_hint="ar")
    assert policy.resolve("ezay a7seb el mesaha?").code == "ar"


def test_a_refused_decision_is_not_verified(policy):
    decision = policy.resolve("ok")
    assert decision.refused is True
    assert decision.verifiable is False


def test_text_without_letters_keeps_continuity(policy):
    """"2 + 2 = ?" says nothing about a language, so the conversation's own
    answer has to carry — which is the session store's job, not a tier nothing
    populated."""
    assert policy.resolve("2 + 2 = ?", session_language="ar").code == "ar"


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


# --- languages we do not answer in ------------------------------------------

@pytest.mark.parametrize("text,detected", [
    ("explícame el ciclo del agua por favor", "es"),
    ("spiegami il ciclo dell'acqua per favore", "it"),
    ("bana su döngüsünü açıkla lütfen", "tr"),
    ("leg me de waterkringloop uit alsjeblieft", "nl"),
    ("explique-me o ciclo da água por favor", "pt"),
])
def test_an_unsupported_language_falls_back_instead_of_guessing(policy, text, detected):
    """A detector built only from the languages we answer in has to pick the
    nearest of them, and does so confidently — Turkish scored 1.00 for German,
    Dutch 0.82. Naming the languages we merely recognise lets us decline."""
    from shared.language.detect import detect

    assert detect(text, SUPPORTED).code == detected
    decision = policy.resolve(text)
    assert decision.code == "en"
    assert decision.refused is True, "a declined language must not trigger repair"


def test_the_caller_s_language_catches_an_unsupported_message(policy):
    """Falling back is where the parameter earns its place."""
    decision = policy.resolve("spiegami il ciclo dell'acqua", api_language="ar")
    assert (decision.code, decision.source) == ("ar", "api")


@pytest.mark.parametrize("text,expected", [
    ("explain column subtraction", "en"),
    ("explique-moi le cycle de l'eau s'il te plaît", "fr"),
    ("erkläre mir bitte den Wasserkreislauf", "de"),
    ("اشرح لي عملية الطرح", "ar"),
])
def test_widening_the_detector_did_not_cost_the_supported_ones(policy, text, expected):
    assert policy.resolve(text).code == expected


# --- refusing a language ----------------------------------------------------

@pytest.mark.parametrize("text", [
    "stop replying in Arabic",
    "don't answer in French",
    "do not reply in Arabic",
    "no longer answer in German",
    "بلاش ترد بالعربي",
])
def test_refusing_a_language_is_not_requesting_it(text):
    """The anchors include reply/respond/answer, so without a negation guard the
    escape hatch reads as a request for the very language being refused — and
    once an override sticks, the student trying to get out locks themselves in."""
    assert explicit_request(text) is None
    assert clear_request(text) is not None


def test_a_negation_does_not_reach_across_a_clause(policy):
    """"I don't like maths" must not cancel the request that follows it."""
    assert policy.resolve("I don't like maths, explain in Arabic").code == "ar"


@pytest.mark.parametrize("text", [
    "answer in the language I write in",
    "reply in my own language please",
    "match my language",
    "رد بنفس لغتي",
    "réponds dans ma langue",
])
def test_asking_to_be_followed_is_a_clear(text):
    assert clear_request(text) is not None


@pytest.mark.parametrize("text", ["reply in Arabic", "explain subtraction", ""])
def test_an_ordinary_message_clears_nothing(text):
    assert clear_request(text) is None


# --- a message that is only an instruction ----------------------------------

@pytest.mark.parametrize("text,expected", [
    ("reply in Arabic", "ar"),
    ("reply in Arabic please", "ar"),
    ("switch to Arabic", "ar"),
    ("من فضلك رد بالعربي", "ar"),
])
def test_a_bare_instruction_is_recognised(text, expected):
    found = language_instruction_only(text)
    assert found is not None and found[1] == expected


@pytest.mark.parametrize("text", [
    "stop replying in Arabic",
    "answer in the language I write in",
])
def test_a_bare_clear_is_recognised(text):
    found = language_instruction_only(text)
    assert found is not None and found[1] is None


@pytest.mark.parametrize("text", [
    "explain subtraction, reply in Arabic",
    "explain subtraction. reply in Arabic",
    "اشرح لي الدرس بالعربي",
    "explain subtraction",
])
def test_a_question_carrying_an_instruction_still_needs_an_answer(text):
    """Both halves matter: the instruction is obeyed, and the question answered."""
    assert language_instruction_only(text) is None


# --- a standing request from earlier in the session --------------------------

@pytest.mark.parametrize("message", [
    "and what about addition?",      # plain English, the common next turn
    "explique-moi la soustraction",  # a confidently-detected other language
    "ok",                            # too short to read
    "2 + 2 = ?",                     # no letters at all
])
def test_a_standing_request_outranks_the_message(policy, message):
    """"Reply in Arabic from now on" means from now on. Only a newer request
    changes it — not the language of the next thing they happen to type."""
    decision = policy.resolve(message, session_language="ar")
    assert (decision.code, decision.source) == ("ar", "session")


def test_a_standing_request_is_verified_not_guessed(policy):
    """These are the turns most likely to drift: the message and the answer are
    in different languages, so the retrieved material pulls against the answer."""
    decision = policy.resolve("and what about addition?", session_language="ar")
    assert decision.refused is False and decision.verifiable is True


def test_a_new_request_replaces_the_standing_one(policy):
    decision = policy.resolve("explain this, reply in French", session_language="ar")
    assert (decision.code, decision.source) == ("fr", "explicit")


def test_the_standing_request_outranks_the_caller(policy):
    decision = policy.resolve("and what about addition?",
                              session_language="ar", api_language="en")
    assert decision.code == "ar"


def test_an_unsupported_standing_value_is_ignored(policy):
    decision = policy.resolve("explain subtraction", session_language="es")
    assert (decision.code, decision.source) == ("en", "detected")


def test_no_standing_request_leaves_everything_as_it_was(policy):
    assert policy.resolve("explain subtraction").code == "en"
    assert policy.resolve("اشرح لي عملية الطرح").code == "ar"


# --- a message about a language, not in one ---------------------------------

@pytest.mark.parametrize("message", [
    "how do you say hello in Arabic?",
    "how to say thank you in French",
    "wie sagt man Hallo auf Arabisch",
    "comment dit-on bonjour en arabe",
])
def test_a_planner_hint_cannot_turn_a_question_about_a_language_into_one_in_it(
    policy, message,
):
    """"how do you say hello in Arabic?" came back in Arabic, explaining the word
    أهلاً out of a poetry lesson.

    explicit_request has always refused to read that as a request. The planner
    reads the same message, is told to report a language only when one is
    explicitly asked for, and reported Arabic anyway — and its hint reached the
    resolver without the test the matcher applies. One question, decided two
    ways in two places."""
    assert policy.resolve(message, planner_hint="ar").code != "ar"


@pytest.mark.parametrize("message,expected", [
    ("ازاي اقول مرحبا بالانجليزي؟", "ar"),
    ("كيف أقول شكرا بالفرنسية", "ar"),
])
def test_the_same_holds_for_a_student_writing_in_arabic(policy, message, expected):
    """The pilot corpus is Arabic, so the Arabic form of the question is the one
    a real student types."""
    assert policy.resolve(message, planner_hint="en").code == expected


def test_asking_for_a_translation_is_still_a_request(policy):
    """The guard must not swallow a real instruction: "translate hello into
    Arabic" genuinely asks for Arabic."""
    assert policy.resolve("translate hello into Arabic").code == "ar"
    assert policy.resolve("what does مرحبا mean in English?").code == "en"


# --- which way a page is read -----------------------------------------------

def test_a_page_of_latin_prose_is_read_left_to_right():
    """reading_direction was one global config value, set to rtl for the pilot
    corpus. An English or French textbook parsed under it would have every
    column read from the wrong side."""
    from shared.language.script import reading_direction

    assert reading_direction("The water cycle moves water between sea and sky.") == "ltr"
    assert reading_direction("Le cycle de l'eau déplace l'eau entre la mer et le ciel.") == "ltr"
    assert reading_direction("Der Wasserkreislauf bewegt Wasser zwischen Meer und Himmel.") == "ltr"


def test_a_page_of_right_to_left_script_is_read_right_to_left():
    """Arabic is not the only right-to-left script, and the pilot corpus being
    Arabic is not a reason for the others to be read backwards."""
    from shared.language.script import reading_direction

    assert reading_direction("جابر بن حيان عالم عربي اشتهر في الكيمياء") == "rtl"      # Arabic
    assert reading_direction("מחזור המים מעביר מים בין הים לשמיים") == "rtl"           # Hebrew
    assert reading_direction("آب کا چکر سمندر اور آسمان کے درمیان پانی منتقل کرتا ہے") == "rtl"  # Urdu


def test_a_page_of_numbers_and_diagrams_falls_back_to_left_to_right():
    """A maths page may carry no letters at all. Left to right is the safe
    default: it is what every Latin, Cyrillic and CJK corpus needs, and a page
    with nothing to go on should not be treated as the exception."""
    from shared.language.script import reading_direction

    assert reading_direction("81 - 52 = 29") == "ltr"
    assert reading_direction("") == "ltr"


def test_a_page_is_read_by_what_most_of_it_is_written_in():
    """A page of Arabic prose with an English caption is still an Arabic page."""
    from shared.language.script import reading_direction

    mixed = "جابر بن حيان عالم عربي اشتهر في الكيمياء. Figure 1: the alembic."
    assert reading_direction(mixed) == "rtl"
