from __future__ import annotations

from ingestion.layout.blocks import Block
from ingestion.utils.normalise import normalise_text


def test_a_text_layer_with_nul_bytes_is_stored_without_them():
    """A Geometry textbook's text layer carried NUL bytes; Postgres refused the
    chunks and two of its lessons failed to store."""
    block = Block(text="Inductive\x00 Reasoning\x07", type="heading", bbox=(0, 0, 1, 1))
    assert block.text == "Inductive Reasoning"
    assert normalise_text("If a = b,\x00 then b = a.\nNext\x1f line") == "If a = b, then b = a.\nNext line"
