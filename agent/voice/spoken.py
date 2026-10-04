"""A tutor reply as something a voice can say.

The reply is written to be read: Markdown, LaTeX, citation markers, links and
the symbols that head its trailing lines. Handed to a voice as it is, a student
hears "dollar x caret two dollar" and "open bracket one close bracket". This
turns it into the words a teacher would say, in the language the reply is in.

Deterministic on purpose. A model asked to rewrite the reply for speech could
also change what it says, and the text a student hears must be the text they
were shown.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ..citations import MARKER, RUN

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$|\\\((.+?)\\\)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"(?:https?://|s3://|www\.)\S+", re.IGNORECASE)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-*+•]\s+", re.MULTILINE)
_RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~|`)(?=\S)(.+?)(?<=\S)\1")
_TRAILER = ("📚", "🧩")
_SPACES = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([.,;:!?؟،])")
_BLANK_LINES = re.compile(r"\n{2,}")


@dataclass(frozen=True)
class MathWords:
    over: str
    squared: str
    cubed: str
    power: str
    root: str
    nth_root: str
    times: str
    divided: str
    plus: str
    minus: str
    plus_minus: str
    equals: str
    not_equal: str
    less: str
    greater: str
    less_equal: str
    greater_equal: str
    approx: str
    infinity: str
    degrees: str
    percent: str
    greek: dict[str, str] = field(default_factory=dict)


MATH_WORDS: dict[str, MathWords] = {
    "en": MathWords(over="over", squared="squared", cubed="cubed",
                    power="to the power of", root="the square root of",
                    nth_root="root {n} of", times="times", divided="divided by",
                    plus="plus", minus="minus", plus_minus="plus or minus",
                    equals="equals", not_equal="is not equal to", less="is less than",
                    greater="is greater than", less_equal="is less than or equal to",
                    greater_equal="is greater than or equal to", approx="is about",
                    infinity="infinity", degrees="degrees", percent="percent"),
    "ar": MathWords(over="على", squared="تربيع", cubed="تكعيب", power="أس",
                    root="الجذر التربيعي لـ", nth_root="الجذر {n} لـ", times="في",
                    divided="مقسومًا على", plus="زائد", minus="ناقص",
                    plus_minus="زائد أو ناقص", equals="يساوي", not_equal="لا يساوي",
                    less="أصغر من", greater="أكبر من", less_equal="أصغر من أو يساوي",
                    greater_equal="أكبر من أو يساوي", approx="تقريبًا",
                    infinity="ما لا نهاية", degrees="درجة", percent="في المئة",
                    greek={"alpha": "ألفا", "beta": "بيتا", "gamma": "جاما",
                           "delta": "دلتا", "Delta": "دلتا", "epsilon": "إبسيلون",
                           "theta": "ثيتا", "lambda": "لامدا", "mu": "ميو", "pi": "باي",
                           "rho": "رو", "sigma": "سيجما", "Sigma": "سيجما", "tau": "تاو",
                           "phi": "فاي", "omega": "أوميجا", "Omega": "أوميجا"}),
    "fr": MathWords(over="sur", squared="au carré", cubed="au cube",
                    power="puissance", root="racine carrée de", nth_root="racine {n} de",
                    times="fois", divided="divisé par", plus="plus", minus="moins",
                    plus_minus="plus ou moins", equals="égale", not_equal="est différent de",
                    less="est inférieur à", greater="est supérieur à",
                    less_equal="est inférieur ou égal à",
                    greater_equal="est supérieur ou égal à", approx="vaut environ",
                    infinity="l'infini", degrees="degrés", percent="pour cent"),
    "de": MathWords(over="durch", squared="hoch zwei", cubed="hoch drei", power="hoch",
                    root="die Quadratwurzel aus", nth_root="die {n}-te Wurzel aus",
                    times="mal", divided="geteilt durch", plus="plus", minus="minus",
                    plus_minus="plus minus", equals="gleich", not_equal="ungleich",
                    less="kleiner als", greater="größer als",
                    less_equal="kleiner oder gleich", greater_equal="größer oder gleich",
                    approx="ungefähr", infinity="unendlich", degrees="Grad",
                    percent="Prozent"),
}

_GREEK = ("alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "pi",
          "rho", "sigma", "tau", "phi", "omega", "Delta", "Sigma", "Omega")
_DROPPED_COMMANDS = re.compile(r"\\(?:left|right|displaystyle|,|;|!|quad|qquad)\b|\\[,;!]")
_TEXT_COMMAND = re.compile(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}")
_FRACTION = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_NTH_ROOT = re.compile(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}")
_ROOT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_DEGREE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\degree|°")
_POWER = re.compile(r"\^\s*(\{[^{}]*\}|\S)")
_SUBSCRIPT = re.compile(r"_\s*(\{[^{}]*\}|\S)")


def speakable(text: str, language: str) -> str:
    """``text`` with everything a voice should not read turned into words or removed."""
    words = MATH_WORDS.get(language, MATH_WORDS["en"])
    text = _FENCE.sub(" ", text or "")
    text = _without_trailers(text)
    text = _DISPLAY_MATH.sub(lambda m: f" {verbalize(m.group(1) or m.group(2), words)} ", text)
    text = _INLINE_MATH.sub(lambda m: verbalize(m.group(1) or m.group(2), words), text)
    text = RUN.sub(" ", text)
    text = MARKER.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(lambda m: m.group(1), text)
    text = _BARE_URL.sub(" ", text)
    text = _TABLE_DIVIDER.sub("", text)
    text = "\n".join(_table_row(line) for line in text.splitlines())
    text = _RULE.sub("", text)
    text = _HEADING.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub("", text)
    for _ in range(3):
        text = _EMPHASIS.sub(lambda m: m.group(2), text)
    text = _without_symbols(text)
    lines = [_SPACE_BEFORE_PUNCTUATION.sub(r"\1", _SPACES.sub(" ", line)).strip()
             for line in text.splitlines()]
    return _BLANK_LINES.sub("\n", "\n".join(lines)).strip()


def verbalize(latex: str, words: MathWords) -> str:
    """One LaTeX expression as words."""
    expr = _TEXT_COMMAND.sub(lambda m: f" {m.group(1)} ", latex)
    expr = _DROPPED_COMMANDS.sub(" ", expr)
    for _ in range(4):
        expr = _FRACTION.sub(lambda m: f" {m.group(1)} {words.over} {m.group(2)} ", expr)
        expr = _NTH_ROOT.sub(
            lambda m: f" {words.nth_root.format(n=m.group(1).strip())} {m.group(2)} ", expr)
        expr = _ROOT.sub(lambda m: f" {words.root} {m.group(1)} ", expr)
    expr = _DEGREE.sub(f" {words.degrees} ", expr)
    expr = _POWER.sub(lambda m: f" {_power(_unbrace(m.group(1)), words)} ", expr)
    expr = _SUBSCRIPT.sub(lambda m: f" {_unbrace(m.group(1))} ", expr)
    for command, spoken in (("\\pm", words.plus_minus), ("\\times", words.times),
                            ("\\cdot", words.times), ("\\div", words.divided),
                            ("\\neq", words.not_equal), ("\\ne", words.not_equal),
                            ("\\leq", words.less_equal), ("\\le", words.less_equal),
                            ("\\geq", words.greater_equal), ("\\ge", words.greater_equal),
                            ("\\approx", words.approx), ("\\infty", words.infinity),
                            ("\\%", words.percent)):
        expr = re.sub(re.escape(command) + r"(?![A-Za-z])", f" {spoken} ", expr)
    for letter in _GREEK:
        expr = re.sub(rf"\\{letter}(?![A-Za-z])", f" {words.greek.get(letter, letter)} ", expr)
    for symbol, spoken in (("<=", words.less_equal), (">=", words.greater_equal),
                           ("=", words.equals), ("+", words.plus), ("<", words.less),
                           (">", words.greater), ("%", words.percent), ("/", words.over)):
        expr = expr.replace(symbol, f" {spoken} ")
    expr = re.sub(r"(?<![\w])-(?=\s*[\w(])|(?<=[\w)\s])-(?=[\s\w(])", f" {words.minus} ", expr)
    expr = expr.replace("*", f" {words.times} ")
    expr = re.sub(r"\\([A-Za-z]+)", r" \1 ", expr)
    expr = re.sub(r"[{}\\]", " ", expr)
    return _SPACES.sub(" ", expr).strip()


def _power(exponent: str, words: MathWords) -> str:
    if exponent == "2":
        return words.squared
    if exponent == "3":
        return words.cubed
    return f"{words.power} {exponent}"


def _unbrace(value: str) -> str:
    value = value.strip()
    return value[1:-1].strip() if value.startswith("{") and value.endswith("}") else value


def _without_trailers(text: str) -> str:
    """The reply without its sources and prerequisites blocks: each runs from its
    symbol to the next blank line, and is for the eye, not the ear."""
    kept, skipping = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(_TRAILER):
            skipping = True
        elif skipping and not line.strip():
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _table_row(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        return ", ".join(cell for cell in cells if cell)
    return line


def _without_symbols(text: str) -> str:
    return "".join(" " if unicodedata.category(ch) == "So" or ch in "→←↑↓|" else ch
                   for ch in text if ch not in "\ufe0f\u200d")
