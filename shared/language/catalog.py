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
    "intent.greeting": {
        "ar": "أهلًا بك! أنا مساعدك في مذاكرة منهجك. اسألني عن أي درس أو فكرة لم تتضح لك.",
        "en": "Hello! I'm your study assistant for your syllabus. Ask me about any lesson or idea you'd like explained.",
        "fr": "Bonjour ! Je suis ton assistant d'étude pour ton programme. Pose-moi une question sur n'importe quelle leçon.",
        "de": "Hallo! Ich bin deine Lernhilfe für deinen Lehrplan. Frag mich zu jeder Lektion oder Idee.",
    },
    "intent.thanks": {
        "ar": "العفو! إن احتجت شرحًا آخر أو تمارين على الدرس، أنا هنا.",
        "en": "You're welcome! If you'd like another explanation or some practice questions, just ask.",
        "fr": "Avec plaisir ! Si tu veux une autre explication ou des exercices, demande-moi.",
        "de": "Gern geschehen! Wenn du noch eine Erklärung oder Übungen möchtest, sag Bescheid.",
    },
    "intent.meta": {
        "ar": "أستطيع أن أشرح دروس منهجك، وألخّصها، وأضع لك أسئلة تدريبية، وأصحّح إجابتك، وأرسم لك خرائط ذهنية أو رسومًا توضيحية. اسألني عن أي درس.",
        "en": "I can explain lessons from your syllabus, summarise them, set practice questions, mark your answers, and draw mind maps or diagrams. Ask me about any lesson.",
        "fr": "Je peux expliquer les leçons de ton programme, les résumer, créer des exercices, corriger tes réponses et dessiner des cartes mentales. Pose-moi une question sur une leçon.",
        "de": "Ich kann Lektionen aus deinem Lehrplan erklären, zusammenfassen, Übungsfragen stellen, deine Antworten bewerten und Mindmaps zeichnen. Frag mich zu einer Lektion.",
    },
    "intent.language": {
        "ar": "تمام، سأجيبك بالعربية من الآن فصاعدًا. عن أي درس تحب أن تسأل؟",
        "en": "Got it — I'll reply in English from now on. Which lesson would you like to ask about?",
        "fr": "C'est noté — je te répondrai en français à partir de maintenant. Sur quelle leçon veux-tu travailler ?",
        "de": "Alles klar — ich antworte ab jetzt auf Deutsch. Zu welcher Lektion hast du eine Frage?",
    },
    "intent.language_cleared": {
        "ar": "تمام، سأردّ بنفس اللغة التي تكتب بها. عن أي درس تحب أن تسأل؟",
        "en": "Got it — I'll reply in whichever language you write in. Which lesson would you like to ask about?",
        "fr": "C'est noté — je répondrai dans la langue que tu utilises. Sur quelle leçon veux-tu travailler ?",
        "de": "Alles klar — ich antworte in der Sprache, in der du schreibst. Zu welcher Lektion hast du eine Frage?",
    },
    "intent.chitchat": {
        "ar": "بخير، شكرًا لسؤالك! هيّا نذاكر — عن أي درس تحب أن تسأل؟",
        "en": "I'm well, thank you for asking! Let's get to studying — which lesson would you like to ask about?",
        "fr": "Je vais bien, merci ! Passons à l'étude — sur quelle leçon veux-tu travailler ?",
        "de": "Mir geht es gut, danke der Nachfrage! Lass uns lernen — zu welcher Lektion hast du eine Frage?",
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
    "explain.out_of_scope": {
        "ar": "نحن ندرس الآن درس «{lesson}». اسألني عن هذا الدرس.",
        "en": "We're studying **{lesson}** right now. Ask me about this lesson.",
        "fr": "Nous étudions **{lesson}** en ce moment. Pose-moi une question sur cette leçon.",
        "de": "Wir behandeln gerade **{lesson}**. Stell mir eine Frage zu dieser Lektion.",
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
    "report.focus_next": {
        "ar": "ركّز بعد ذلك على: {focus}.",
        "en": "Focus next on: {focus}.",
        "fr": "Concentre-toi ensuite sur : {focus}.",
        "de": "Konzentriere dich als Nächstes auf: {focus}.",
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


