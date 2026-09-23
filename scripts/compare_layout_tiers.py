"""Read the same pages by each parser we have, and count what the index gets.

Ingesting a book is a decision about which parser reads it, and until now that
decision was made from a comment in the extractor rather than from a number.
This reads a slice of a PDF both ways and reports what the chunker would write
to the index from each, so a book can be judged before it is ingested.

    python scripts/compare_layout_tiers.py book.pdf --pages 16-25

What the numbers mean:

  chunks          rows the index would gain from this slice
  median tokens   a chunk far under ~100 tokens is a fragment that matches
                  nothing; far over 800 averages too many topics to match well
  mid-sentence    chunks that end without a sentence ending: text cut mid-thought
  salad           chunks whose words are mostly one or two characters long, the
                  signature of a diagram read as if it were prose
  furniture       chunks still carrying a running head, folio or credit line
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402

from ingestion.layout import parse_page, tier2, tier3  # noqa: E402
from ingestion.layout.blocks import PageMeta, ParsedPage  # noqa: E402
from ingestion.layout.chunker import page_to_chunks  # noqa: E402
from ingestion.utils.tokens import count_tokens  # noqa: E402
from shared.config import get_config  # noqa: E402

SENTENCE_END = re.compile(r"[.!?؟۔।]['\")\]]?\s*$")
FURNITURE = re.compile(r"(?i)copyright|all rights reserved|houghton|©")


def pages(spec: str, total: int) -> list[int]:
    if "-" in spec:
        first, last = (int(n) for n in spec.split("-", 1))
        return [p for p in range(first - 1, last) if 0 <= p < total]
    return [int(n) - 1 for n in spec.split(",") if 0 <= int(n) - 1 < total]


def measure(chunks) -> dict:
    if not chunks:
        return {"chunks": 0, "median": 0, "mid_sentence": 0, "salad": 0, "furniture": 0}
    tokens = [count_tokens(c.text) for c in chunks]
    return {
        "chunks": len(chunks),
        "median": int(statistics.median(tokens)),
        "mid_sentence": sum(1 for c in chunks if not SENTENCE_END.search(c.text.strip())),
        "salad": sum(1 for c in chunks if _salad(c.text)),
        "furniture": sum(1 for c in chunks if FURNITURE.search(c.text)),
    }


def _salad(text: str) -> bool:
    words = text.split()
    if len(words) < 8:
        return False
    return sum(1 for w in words if len(w) <= 2) / len(words) > 0.5


def read(page, meta: PageMeta, tier: str, *, dpi: int, config,
         language: str | None) -> ParsedPage | None:
    if tier == "text":
        return parse_page(page, meta, direction=meta.reading_direction)
    rendered = page.get_pixmap(dpi=dpi).tobytes("png")
    if tier == "layout":
        return tier2.parse_page(rendered, meta, config=config)
    return tier3.parse_page(rendered, meta, config=config, language_hint=language,
                            direction=meta.reading_direction)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--pages", default="1-10", help="1-based, e.g. 16-25 or 3,9,17")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", type=int, default=0, help="print this many chunks from each")
    parser.add_argument("--tiers", default="text,layout",
                        help="which parsers to run: text, layout, vision (comma separated). "
                             "A scanned book has no text layer, so there it is layout,vision")
    parser.add_argument("--language", default=None, help="language hint for the vision tier")
    args = parser.parse_args()

    config = get_config()
    config.override("ingestion.layout.tier2.enabled", True)
    doc = fitz.open(args.pdf)
    wanted = pages(args.pages, doc.page_count)
    print(f"{Path(args.pdf).name}: {len(wanted)} of {doc.page_count} pages\n")

    labels = {"text": "text layer", "layout": "layout model", "vision": "vision model"}
    results = {}
    for tier in [t.strip() for t in args.tiers.split(",") if t.strip()]:
        label = labels.get(tier, tier)
        chunks, unread, started = [], 0, time.time()
        for index in wanted:
            page = doc[index]
            meta = PageMeta(doc_id="compare", page=index + 1,
                            width=float(page.rect.width), height=float(page.rect.height))
            parsed = read(page, meta, tier, dpi=args.dpi, config=config,
                          language=args.language)
            if parsed is None:
                unread += 1
                continue
            chunks.extend(page_to_chunks(parsed))
        results[label] = {**measure(chunks), "unread": unread,
                          "seconds": round(time.time() - started, 1)}
        if args.show:
            print(f"--- {label}, first {args.show} chunks")
            for chunk in chunks[:args.show]:
                print(f"  [{chunk.heading or '-'}] {chunk.text[:160]}")
            print()

    width = max(len(k) for k in results)
    header = f"{'':<{width}}  chunks  median  mid-sent   salad  furniture  unread  secs"
    print(header)
    for label, row in results.items():
        print(f"{label:<{width}}  {row['chunks']:>6}  {row['median']:>6}  "
              f"{row['mid_sentence']:>8}  {row['salad']:>6}  {row['furniture']:>9}  "
              f"{row['unread']:>6}  {row['seconds']:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
