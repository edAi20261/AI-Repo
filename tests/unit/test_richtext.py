"""Lesson body text — the platform editor stores HTML, the seed scripts Markdown."""

from __future__ import annotations

from ingestion.utils.richtext import to_blocks

EDITOR_HTML = (
    "<p><strong>What is Addition?</strong></p>"
    "<ul><li><p>Putting together: combining quantities.</p></li>"
    "<li><p>Counting on: start from the larger number.</p></li></ul>"
)


def test_html_paragraphs_and_list_items_become_separate_blocks():
    blocks = to_blocks(EDITOR_HTML)

    assert [b.text for b in blocks] == [
        "What is Addition?",
        "Putting together: combining quantities.",
        "Counting on: start from the larger number.",
    ]
    assert not any(b.is_heading for b in blocks)


def test_html_headings_carry_their_depth():
    blocks = to_blocks("<h1>Unit 2</h1><p>Intro</p><h2>Lesson 1</h2><p>Body</p>")

    assert [(b.text, b.level) for b in blocks] == [
        ("Unit 2", 1),
        ("Intro", 0),
        ("Lesson 1", 2),
        ("Body", 0),
    ]


def test_markdown_headings_carry_their_depth():
    blocks = to_blocks("# Lab Safety\n\nWear goggles at all times.\n\n## Before you start\n\nCheck.")

    assert [(b.text, b.level) for b in blocks] == [
        ("Lab Safety", 1),
        ("Wear goggles at all times.", 0),
        ("Before you start", 2),
        ("Check.", 0),
    ]


def test_markdown_paragraph_lines_join_into_one_block():
    blocks = to_blocks("The plant uses light\nand stores energy.\n\nSecond paragraph.")

    assert [b.text for b in blocks] == [
        "The plant uses light and stores energy.",
        "Second paragraph.",
    ]


def test_entities_and_whitespace_are_normalised():
    blocks = to_blocks("<p>Water   &amp;   air\n\tmix</p>")

    assert blocks[0].text == "Water & air mix"


def test_script_and_style_content_is_dropped():
    blocks = to_blocks("<style>p{color:red}</style><p>Visible</p><script>x=1</script>")

    assert [b.text for b in blocks] == ["Visible"]


def test_line_breaks_split_blocks():
    blocks = to_blocks("<p>First line<br/>Second line</p>")

    assert [b.text for b in blocks] == ["First line", "Second line"]


def test_arabic_html_survives_unchanged():
    blocks = to_blocks("<h2>الدرس الأول</h2><p>ريادة الأعمال</p>")

    assert [(b.text, b.level) for b in blocks] == [("الدرس الأول", 2), ("ريادة الأعمال", 0)]


def test_empty_input_yields_nothing():
    assert to_blocks("") == []
    assert to_blocks("   \n  ") == []
    assert to_blocks("<p></p><div>  </div>") == []
