"""Vision prompt: full-page maths transcription (text + figures), fidelity-focused.

Shares the JSON contract of the general page prompt but insists on mathematical
fidelity (fractions, roots, column order).
"""

from .page_transcription import _JSON_CONTRACT

MATH_PAGE_TRANSCRIPTION_PROMPT = (
    "You are transcribing one page of a mathematics textbook and locating its "
    "figures. Mathematical fidelity is the priority:\n"
    "- Every fraction as (numerator)/(denominator): 8/12, never stacked onto two "
    "lines.\n"
    "- Preserve = signs, roots (sqrt), exponents (^), subscripts (_), and the "
    "top-to-bottom order of each worked step.\n"
    "- In multi-column layouts, finish one column before the next — never read "
    "across columns and interleave two worked examples.\n"
    "- For \"figures\", box ONLY real content figures — geometry diagrams, graphs, "
    "charts, number lines and labelled illustrations. Do NOT box equations, "
    "worked-solution text, tables, decorative banners, badges or rating icons.\n\n"
    + _JSON_CONTRACT
    + "\nLanguage hint (may be empty): {language_hint}\n"
)
