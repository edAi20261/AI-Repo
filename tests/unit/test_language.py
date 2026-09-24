"""Language policy: what a turn is answered in, catalogue coverage and verification.

The named regressions are the failures that shaped this layer: a student asked
"explain column subtraction" and was answered in French, and a student who asked
"why did you answer in French?" was answered in French for the next month.
"""

from __future__ import annotations

import pytest

from shared.language import (
    LanguageDecision,
    LanguagePolicy,
    classify,
    content_directive,
    language_directive,
    looks_arabizi,
    verify_answer,
)
from shared.language.verify import prose_of

SUPPORTED = ("ar", "en", "fr", "de")


@pytest.fixture
def policy() -> LanguagePolicy:
    return LanguagePolicy(supported=SUPPORTED, default="en")


def _reply_to(policy, message, **facts):
    return policy.reply(message_language=policy.written_in(message), **facts)


# --- the reported bugs ------------------------------------------------------

def test_short_english_question_is_not_french(policy):
    """The reported bug: 24 letters of English scored 0.57 for French."""
    assert _reply_to(policy, "explain column subtraction").code == "en"


def test_single_word_greeting_is_not_finnish(policy):
    assert _reply_to(policy, "hello").code == "en"


def test_the_policy_reads_no_words_as_requests(policy):
    """"why did you answer in french?" was matched as a request and stored for a
    month. The policy now ranks facts it is handed; reading a request is the
    model's job, so no phrasing of a message can switch the language here."""
    for message in ("why did you answer in french?", "réponds en français",
                    "is the answer in French correct?"):
        decision = policy.reply(subject_language="en",
                                message_language=policy.written_in(message))
        assert decision.code == "en"
        assert decision.source == "subject"


# --- precedence -------------------------------------------------------------

def test_a_request_the_student_made_beats_everything(policy):
    decision = policy.reply(requested="fr", subject_language="ar",
                            message_language="en", api_language="de")
    assert (decision.code, decision.source) == ("fr", "session")


def test_a_subject_is_taught_in_its_own_language(policy):
    """An Arabic-speaking student in an English science lesson is answered in
    English unless they ask otherwise."""
    decision = policy.reply(subject_language="en", message_language="ar")
    assert (decision.code, decision.source) == ("en", "subject")


def test_without_a_subject_the_student_is_answered_in_their_language(policy):
    decision = policy.reply(message_language="ar", api_language="en")
    assert (decision.code, decision.source) == ("ar", "message")


def test_a_deployment_can_answer_in_the_student_s_language_instead():
    policy = LanguagePolicy(supported=SUPPORTED, reply_follows="message")
    decision = policy.reply(subject_language="en", message_language="ar")
    assert (decision.code, decision.source) == ("ar", "message")


def test_the_caller_s_parameter_decides_only_what_nothing_else_can(policy):
    assert policy.reply(api_language="de").code == "de"
    assert policy.reply(message_language="ar", api_language="de").code == "ar"


def test_the_last_resort_is_the_default_and_is_never_repaired(policy):
    decision = policy.reply()
    assert (decision.code, decision.source) == ("en", "default")
    assert decision.verifiable is False


def test_unsupported_values_are_ignored(policy):
    decision = policy.reply(requested="it", subject_language="es", message_language="tr")
    assert decision.source == "default"


# --- reading the message ----------------------------------------------------

def test_script_decides_arabic(policy):
    assert policy.written_in("ما هي الجملة الاسمية؟") == "ar"


@pytest.mark.parametrize("text", [
    "momken tefhemni el dars da?",
    "ezay a7seb el mesaha?",
    "ok",
])
def test_what_the_text_cannot_tell_is_left_undecided(policy, text):
    """Arabizi and two-letter replies defeat every detector measured; the model
    reading the conversation decides these, and the policy never guesses."""
    assert policy.written_in(text) is None


@pytest.mark.parametrize("text,expected", [
    ("explain column subtraction", "en"),
    ("explique-moi le cycle de l'eau s'il te plaît", "fr"),
    ("erkläre mir bitte den Wasserkreislauf", "de"),
    ("اشرح لي عملية الطرح", "ar"),
])
def test_widening_the_detector_did_not_cost_the_supported_ones(policy, text, expected):
    assert _reply_to(policy, text).code == expected


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
    assert policy.written_in(text) is None
    assert _reply_to(policy, text).code == "en"


# --- script -----------------------------------------------------------------

def test_script_classification():
    assert classify("ما هي الجملة الاسمية؟") == "arabic"
    assert classify("what is a noun") == "latin"
    assert classify("a7seb el mesaha") == "arabizi"
    assert classify("") == "none"


def test_arabizi_needs_evidence():
    assert looks_arabizi("a7seb") is True
    assert looks_arabizi("hello there") is False


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



# --- reading direction ------------------------------------------------------

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


# --- the same subject in two languages ---------------------------------------

def test_a_student_writing_arabic_is_not_answered_in_french():
    """A term now exists twice, in Arabic and in French. A student asked
    'ممكن تشرح لي الكهربية الساكنة؟' and was answered in French, because the
    conversation had settled on the French subject two turns earlier and the
    reply language followed the subject rather than the student."""
    policy = LanguagePolicy(supported=("ar", "en", "fr"), default="en")

    decision = policy.reply(subject_language="fr", message_language="ar",
                            subject_chosen=False)
    assert decision.code == "ar"
    assert decision.source == "message"


def test_an_opened_lesson_still_teaches_in_its_own_language():
    """The rule this does not overturn: a student inside an English literature
    lesson is answered in English, whatever language they type in."""
    policy = LanguagePolicy(supported=("ar", "en", "fr"), default="en")

    decision = policy.reply(subject_language="en", message_language="ar",
                            subject_chosen=True)
    assert decision.code == "en"
    assert decision.source == "subject"


def test_asking_for_a_language_still_beats_everything():
    policy = LanguagePolicy(supported=("ar", "en", "fr"), default="en")
    for chosen in (True, False):
        decision = policy.reply(requested="fr", subject_language="ar",
                                message_language="ar", subject_chosen=chosen)
        assert decision.code == "fr" and decision.source == "session"


# --- the rule is written in the language it governs ---------------------------

def test_an_answer_is_told_not_to_greet_in_the_language_it_answers_in():
    """16 of 40 production replies opened with a greeting, and every one of them
    was Arabic or French — the English replies opened with the answer. The rule
    was in English while the model composed in Arabic, where أهلاً بك is simply
    how a message starts."""
    from shared.language.directive import language_directive

    arabic = language_directive("ar", greeting=False)
    assert "أهلا" in arabic or "ترحيب" in arabic

    french = language_directive("fr", greeting=False)
    assert "Bonjour" in french or "salutation" in french.lower()

    english = language_directive("en", greeting=False)
    assert "greeting" in english.lower()


def test_a_social_turn_keeps_its_greeting():
    """A student who says hello is answered by the converse prompt, and that one
    should say hello back."""
    from shared.language.directive import language_directive

    assert "greeting" not in language_directive("en").lower()


def test_every_supported_language_has_the_rule_in_its_own_words():
    """An English sentence inside an Arabic prompt is the defect, not the fix."""
    from shared.language.directive import NO_GREETING
    from shared.language.resolver import DEFAULT_SUPPORTED

    for code in DEFAULT_SUPPORTED:
        assert code in NO_GREETING, code
