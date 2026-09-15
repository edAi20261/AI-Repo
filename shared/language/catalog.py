"""Every user-facing string the service writes itself.

Text the model does not generate — refusals, headings, labels — was scattered
across the nodes as ad-hoc dicts, several of which covered only Arabic and
English while four languages were on offer. Collecting them here makes a missing
translation a test failure instead of an English sentence inside an Arabic reply.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

STRINGS: dict[str, dict[str, str]] = {
    "common.degraded": {
        "ar": "عذرًا، تعذّر إنجاز هذا الآن. من فضلك حاول مرة أخرى بعد قليل.",
        "en": "Sorry — I couldn't complete this right now. Please try again in a moment.",
        "fr": "Désolé — je n'ai pas pu réaliser cela pour le moment. Veuillez réessayer dans un instant.",
        "de": "Entschuldigung — das war gerade nicht möglich. Bitte versuchen Sie es gleich noch einmal.",
    },
    "classify.sign_in": {
        "ar": "عذرًا، يرجى تسجيل الدخول بحساب صالح (طالب أو معلّم أو وليّ أمر) للمتابعة.",
        "en": "Sorry — please sign in with a valid account (student, teacher, or parent) to continue.",
        "fr": "Désolé — veuillez vous connecter avec un compte valide (élève, enseignant ou parent).",
        "de": "Entschuldigung — bitte melden Sie sich mit einem gültigen Konto an (Schüler, Lehrer oder Elternteil).",
    },
    "compose.prereq_head": {
        "ar": "مواضيع تمهيدية", "en": "Prerequisite topics",
        "fr": "Sujets prérequis", "de": "Voraussetzungen",
    },
    "compose.sources_head": {
        "ar": "المصادر", "en": "Sources", "fr": "Sources", "de": "Quellen",
    },
    "compose.lesson_word": {
        "ar": "الدرس", "en": "Lesson", "fr": "Leçon", "de": "Lektion",
    },
    "compose.from_syllabus": {
        "ar": " (من منهجك)", "en": " (from your syllabus)",
        "fr": " (de ton programme)", "de": " (aus deinem Lehrplan)",
    },
    "compose.no_material": {
        "ar": "لا توجد مادة كافية في المنهج للإجابة.",
        "en": "There isn't enough syllabus material to answer this yet.",
        "fr": "Il n'y a pas assez de contenu du programme pour répondre à cela.",
        "de": "Es gibt noch nicht genug Lehrplaninhalt, um dies zu beantworten.",
    },
    "compose.assessment_line": {
        "ar": "إليك اختبارًا من {n} سؤالًا.",
        "en": "Here's a {n}-question assessment.",
        "fr": "Voici une évaluation de {n} questions.",
        "de": "Hier ist eine Bewertung mit {n} Fragen.",
    },
    "visual.caption": {
        "ar": "إليك {noun}:", "en": "Here's your {noun}:",
        "fr": "Voici votre {noun} :", "de": "Hier ist Ihre {noun}:",
    },
    "visual.render_error": {
        "ar": "حدث خطأ أثناء إنشاء العنصر المرئي. يرجى المحاولة مرة أخرى لاحقًا.",
        "en": "Something went wrong creating the visual. Please try again later.",
        "fr": "Une erreur s'est produite lors de la création du visuel. Veuillez réessayer plus tard.",
        "de": "Beim Erstellen des Visuals ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.",
    },
    "visual.noun.mindmap": {
        "ar": "خريطة ذهنية", "en": "mind map", "fr": "carte mentale", "de": "Mindmap",
    },
    "visual.noun.flashcards": {
        "ar": "بطاقات تعليمية", "en": "flashcards", "fr": "cartes mémoire", "de": "Lernkarten",
    },
    "visual.noun.pdf": {"ar": "ملف PDF", "en": "PDF", "fr": "PDF", "de": "PDF"},
    "visual.noun.chart": {
        "ar": "رسم بياني", "en": "chart", "fr": "graphique", "de": "Diagramm",
    },
    "visual.noun.image": {
        "ar": "رسم توضيحي", "en": "illustration", "fr": "illustration", "de": "Illustration",
    },
    "visual.noun.figure": {"ar": "شكل", "en": "figure", "fr": "figure", "de": "Abbildung"},
    "grade.answer_label": {
        "ar": "الإجابة الصحيحة", "en": "Correct answer",
        "fr": "Bonne réponse", "de": "Richtige Antwort",
    },
    "grade.why_label": {
        "ar": "الشرح", "en": "Why", "fr": "Explication", "de": "Erklärung",
    },
    "grade.head": {
        "ar": "صحّحتُ إجابتك: {score} من {total}.",
        "en": "I've graded your work: {score} / {total}.",
        "fr": "J'ai corrigé ton travail : {score} / {total}.",
        "de": "Ich habe deine Arbeit bewertet: {score} / {total}.",
    },
    "grade.no_evidence": {
        "ar": "لا يوجد نموذج إجابة أو محتوى كافٍ من المنهج لتقييم هذه الإجابة.",
        "en": "There isn't a reference answer or enough syllabus material to grade this answer.",
        "fr": "Aucune réponse de référence ni contenu suffisant du programme pour évaluer cette réponse.",
        "de": "Es gibt keine Musterlösung oder ausreichenden Lehrplaninhalt, um diese Antwort zu bewerten.",
    },
    "explain.no_context": {
        "ar": "لا توجد مادة كافية في المنهج للإجابة عن هذا السؤال.",
        "en": "There isn't enough syllabus material to answer this yet.",
        "fr": "Il n'y a pas encore assez de contenu du programme pour répondre à cette question.",
        "de": "Es gibt noch nicht genug Lehrplaninhalt, um diese Frage zu beantworten.",
    },
    "report.all_good": {
        "ar": "أداؤك جيد — لا توجد نقاط ضعف واضحة حاليًا. واصل التقدّم.",
        "en": "You're on track — no weak spots are showing up right now. Keep it up.",
        "fr": "Tu es sur la bonne voie — aucun point faible pour l'instant. Continue.",
        "de": "Du bist auf einem guten Weg — derzeit keine Schwachstellen. Weiter so.",
    },
    "report.no_data": {
        "ar": "لا يوجد قدر كافٍ من الأعمال المصحّحة بعد لإعداد تقرير عن تقدّمك.",
        "en": "There isn't enough graded work yet to report on your progress.",
        "fr": "Il n'y a pas encore assez de travail noté pour faire le point sur tes progrès.",
        "de": "Es gibt noch nicht genug bewertete Arbeit, um deinen Fortschritt zu berichten.",
    },
    "report.chart_title": {
        "ar": "تقدّم الإتقان", "en": "Mastery over time",
        "fr": "Maîtrise au fil du temps", "de": "Lernfortschritt",
    },
    "report.chart_axis_y": {
        "ar": "الدرجة", "en": "score", "fr": "score", "de": "Punktzahl",
    },
}

DEFAULT_FALLBACK_CHAIN: tuple[str, ...] = ("en",)


def translate(
    key: str,
    language: str,
    *,
    fallback_chain: tuple[str, ...] = DEFAULT_FALLBACK_CHAIN,
    **values: object,
) -> str:
    """The localized string for ``key``, formatted with ``values``.

    A missing translation is logged rather than silently served in another
    language: an English sentence inside an Arabic answer is a defect, and one
    nobody notices unless it is recorded.
    """
    entries = STRINGS.get(key)
    if entries is None:
        logger.error("i18n missing key", extra={"i18n_key": key, "language": language})
        return key

    text: Optional[str] = entries.get(language)
    if text is None:
        for candidate in fallback_chain:
            text = entries.get(candidate)
            if text is not None:
                logger.warning(
                    "i18n missing translation",
                    extra={"i18n_key": key, "language": language, "served": candidate},
                )
                break
    if text is None:
        logger.error("i18n key has no usable translation",
                     extra={"i18n_key": key, "language": language})
        return key

    return text.format(**values) if values else text


def languages_of(key: str) -> set[str]:
    """Which languages ``key`` is translated into — used by the coverage test."""
    return set(STRINGS.get(key, {}))
