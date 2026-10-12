"""One book, ingested by several machines at once.

On one machine a scanned book is bounded by two kinds of work a single 2-core
runner does slowly: model calls (two per scanned page, about 12.6 s each, of
which one machine can only hold so many in flight) and embedding chunks with
BGE-M3 (measured at 2.5 s a 651-token chunk on an idle 2-vCPU machine). A
300-page book is 600 calls of the first and, at six hundred chunks, twenty-five
minutes of the second. Neither depends on the others' results, so both split.

    plan     how many machines each phase gets, from the book's page count
    read     one share of the pages read into the page cache (ai.page_transcriptions)
    prepare  extract + chunk, with every page already read; writes a state file
    embed    one share of the chunks embedded; writes a vectors file
    finish   attach every share's vectors, then topics, dependencies and index

``read`` shares write only the page cache, under the keys the extract stage
reads, so a share that fails costs nothing but time: ``prepare`` reads whatever
pages are still missing itself. ``finish`` refuses vectors that were made from
any other text than the state it is finishing, or that leave a chunk without
one.

State and vector files are pickles because they carry the pipeline's own
objects between jobs of one workflow run, written and read by the same commit.
They are never read from anywhere else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

from shared.logging import setup_logging
from shared.metering import start_meter

from . import get_config
from .books import LessonCatalog, detect_lessons, file_under_lessons, lesson_states
from .context import PipelineContext
from .graph import run_stages
from .lesson import add_lesson_arguments, apply_lesson_overrides, lesson_state, report
from .read_ahead import ReadAheadUnavailable, pages_of_shard, read_ahead
from .stages.embed import attach, embedding_texts
from .utils import pdf

logger = logging.getLogger(__name__)

FORMAT = 1
# A machine reading fewer pages than this spends more of its time starting up
# than reading.
MIN_PAGES_PER_READER = 10


def texts_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(hashlib.sha256(text.encode("utf-8")).digest())
    return digest.hexdigest()


def shard_bounds(total: int, shard: int, of: int) -> tuple[int, int]:
    """``[lo, hi)`` of ``total`` items that share ``shard`` of ``of`` embeds.

    Contiguous, and as even as integers allow: the shares tile the range
    exactly, which is what ``finish`` checks.
    """
    if of < 1 or not 0 <= shard < of:
        raise ValueError(f"shard {shard} of {of} does not exist")
    return (total * shard) // of, (total * (shard + 1)) // of


# --- plan -------------------------------------------------------------------

def plan(pages: int, *, readers: int, embedders: int) -> dict:
    readers = max(1, min(readers, math.ceil(pages / MIN_PAGES_PER_READER)))
    embedders = max(1, embedders)
    return {
        "pages": pages,
        "read": {"include": [{"shard": i, "of": readers} for i in range(readers)]},
        "embed": {"include": [{"shard": i, "of": embedders} for i in range(embedders)]},
    }


def _cmd_plan(args) -> int:
    doc = pdf.open_document(args.source)
    try:
        pages = len(doc)
    finally:
        doc.close()
    out = plan(pages, readers=args.read_shards, embedders=args.embed_shards)
    print(json.dumps(out))
    target = os.environ.get("GITHUB_OUTPUT")
    if args.github_output and target:
        with open(target, "a", encoding="utf-8") as sink:
            sink.write(f"pages={pages}\n")
            sink.write(f"read_matrix={json.dumps(out['read'])}\n")
            sink.write(f"embed_matrix={json.dumps(out['embed'])}\n")
    return 0


# --- read -------------------------------------------------------------------

def _cmd_read(args, config) -> int:
    state = lesson_state(args, config)
    ctx = PipelineContext.build(config)
    doc = pdf.open_document(args.source)
    try:
        total = len(doc)
    finally:
        doc.close()
    pages = pages_of_shard(total, args.shard, args.of)
    meter = start_meter()
    try:
        outcome = read_ahead(args.source, state, ctx, pages=pages,
                             workers=args.workers or None)
    except ReadAheadUnavailable as reason:
        print(f"cannot read ahead: {reason}", file=sys.stderr)
        return 1
    outcome["cost"] = meter.as_dict()
    print(json.dumps({"shard": args.shard, "of": args.of, **outcome}, ensure_ascii=False))
    asked = outcome.get("to_read", 0)
    kept = outcome.get("layout_read", 0) + outcome.get("transcribed", 0)
    if asked and not kept and not outcome.get("already_paid_for"):
        # Every call failed: a credential, a quota or the network, and the same
        # fault waits for the next machine. Stopping here is cheaper than six
        # more machines finding it. Not on a share already read before: what is
        # left there is the pages that failed last time, a re-run of one blank
        # page failing again said nothing about the credential and stopped the
        # book, and prepare reads such a page again itself.
        print("no page could be read; see the log above", file=sys.stderr)
        return 1
    return 0


# --- prepare ----------------------------------------------------------------

def _cmd_prepare(args, config) -> int:
    state = lesson_state(args, config)
    ctx = PipelineContext.build(config)
    meter = start_meter()
    state = run_stages(state, ctx, ("extract", "chunk"))
    if state.get("book_id"):
        state = file_under_lessons(state, detect_lessons(state, config))
        _summarise_lessons(state)
    texts = embedding_texts(state.get("chunks", []), config)
    _dump(args.out, {
        "format": FORMAT,
        "state": state,
        "texts": texts,
        "digest": texts_digest(texts),
        "cost": meter.as_dict(),
    })
    print(json.dumps({"chunks": len(texts), "stats": state.get("stats", {}),
                      "cost": meter.as_dict()}, ensure_ascii=False, default=str))
    return 0


# --- embed ------------------------------------------------------------------

def _cmd_embed(args, config) -> int:
    prepared = _load(args.state)
    texts: list[str] = prepared["texts"]
    lo, hi = shard_bounds(len(texts), args.shard, args.of)
    share = texts[lo:hi]
    dense: list[list[float]] = []
    sparse = None
    dimension = 0
    if share:
        ctx = PipelineContext.build(config)
        dense = ctx.embedder.embed(share)
        dimension = ctx.embedder.dimension
        if config.get("vector_store.hybrid"):
            sparse = ctx.embedder.embed_sparse(share)
    _dump(args.out, {
        "format": FORMAT,
        "digest": prepared["digest"],
        "lo": lo,
        "hi": hi,
        "dense": dense,
        "sparse": sparse,
        "dimension": dimension,
        "model": config.get("embeddings.model"),
    })
    print(json.dumps({"shard": args.shard, "of": args.of, "chunks": hi - lo}))
    return 0


# --- finish -----------------------------------------------------------------

def collect(prepared: dict, shares: list[dict], *, model: str, hybrid: bool):
    """Every chunk's vectors, in chunk order, from the shares that made them.

    Refuses anything that would index a chunk with a vector made from other
    text, by another model, or with no vector at all — each of which indexes
    without an error and is found by nobody.
    """
    total = len(prepared["texts"])
    shares = sorted(shares, key=lambda s: (s["lo"], s["hi"]))
    cursor = 0
    dense: list[list[float]] = []
    sparse: list[dict[int, float]] = []
    dimension = 0
    for share in shares:
        if share.get("format") != FORMAT:
            raise ValueError("a vectors file of another format")
        if share["digest"] != prepared["digest"]:
            raise ValueError("vectors made from different chunks than this state's")
        if share["model"] != model:
            raise ValueError(f"vectors from {share['model']}, this run embeds with {model}")
        if share["lo"] != cursor:
            raise ValueError(f"chunks {cursor}..{share['lo']} have no vectors"
                             if share["lo"] > cursor else
                             f"chunks from {share['lo']} were embedded twice")
        if len(share["dense"]) != share["hi"] - share["lo"]:
            raise ValueError(f"share {share['lo']}..{share['hi']} is incomplete")
        if hybrid and share["hi"] > share["lo"] and share["sparse"] is None:
            raise ValueError("the index is hybrid and a share has no lexical vectors")
        dense.extend(share["dense"])
        if hybrid and share["sparse"]:
            sparse.extend(share["sparse"])
        dimension = share["dimension"] or dimension
        cursor = share["hi"]
    if cursor != total:
        raise ValueError(f"chunks {cursor}..{total} have no vectors")
    return dense, (sparse if hybrid else None), dimension


def _cmd_finish(args, config) -> int:
    prepared = _load(args.state)
    if prepared.get("format") != FORMAT:
        raise ValueError("a state file of another format")
    shares = [_load(path) for path in sorted(Path(args.vectors).glob("*.pkl"))]
    state = prepared["state"]
    hybrid = bool(config.get("vector_store.hybrid"))
    dense, sparse, dimension = collect(prepared, shares,
                                       model=config.get("embeddings.model"), hybrid=hybrid)
    meter = start_meter()
    if state.get("chunks"):
        state = attach(state, config, dense, sparse,
                       dimension=dimension or config.get("embeddings.dimensions"))
    else:
        state.setdefault("stats", {})["embed"] = {"embedded": 0}
    ctx = PipelineContext.build(config)
    if state.get("lessons"):
        state = _finish_book(state, ctx, config)
    else:
        state = run_stages(state, ctx, ("topics", "dependencies", "index"))
    cost = meter.as_dict()
    state.setdefault("stats", {})["cost"] = cost
    state["stats"]["cost_of_prepare"] = prepared.get("cost")
    report(state["lecture_id"], state)
    _summarise(state)
    return 1 if any(r["status"] == "failed" for r in state.get("lesson_results", [])) else 0


_STAGES = ("topics", "dependencies", "index")


def _finish_book(state: dict, ctx: PipelineContext, config) -> dict:
    catalog = LessonCatalog(config)
    catalog.register(subject_id=state["subject_id"], book_title=state.get("lesson_title"),
                     lessons=state["lessons"])
    results: list[dict] = []
    totals = {"chunks_written": 0, "figures_written": 0, "points_upserted": 0}
    topics = edges = 0
    for part in lesson_states(state):
        try:
            done = run_stages(part, ctx, _STAGES)
            status = done.get("ingestion_status") or "ready"
            errors = done.get("errors", [])
        except Exception as error:
            logger.exception("lesson failed", extra={"lecture_id": part["lecture_id"]})
            done, status, errors = part, "failed", [f"{type(error).__name__}: {error}"]
        catalog.finished(part["lecture_id"], status=status, errors=errors,
                         version=part.get("version", 1))
        stats = done.get("stats", {})
        for key in totals:
            totals[key] += stats.get("index", {}).get(key, 0)
        topics += stats.get("topics", {}).get("extracted", 0)
        edges += stats.get("dependencies", {}).get("edges", 0)
        results.append({"lesson_id": part["lecture_id"], "title": part["lesson_title"],
                        "status": status, "errors": errors,
                        "chunks": stats.get("index", {}).get("chunks_written", 0),
                        "topics": stats.get("topics", {}).get("extracted", 0)})
    stale = catalog.stale(state["book_id"], len(state["lessons"]))
    for lesson_id in stale:
        ctx.chunk_store.delete_lecture(lesson_id)
        ctx.vector_store.delete_lecture(lesson_id)
    catalog.archive(stale)
    stats = state.setdefault("stats", {})
    stats["index"] = totals
    stats["topics"] = {"extracted": topics}
    stats["dependencies"] = {"edges": edges}
    stats["lessons"] = {"lessons": len(results), "archived": len(stale)}
    state["lesson_results"] = results
    state["ingestion_status"] = (
        "failed" if any(r["status"] == "failed" for r in results) else "ready")
    return state


def _summarise_lessons(state: dict) -> None:
    lessons = state.get("lessons", [])
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    print(json.dumps({"lessons": lessons}, ensure_ascii=False))
    if not target:
        return
    counts: dict[str, int] = {}
    for chunk in state.get("chunks", []):
        counts[chunk.lecture_id] = counts.get(chunk.lecture_id, 0) + 1
    lines = [f"## {state.get('lesson_title') or state['lecture_id']}: "
             f"{len(lessons)} lessons found", "",
             "| # | Lesson | Unit | Pages | Chunks | Lesson ID |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {lsn['title']} | {lsn['unit_title'] or lsn['unit_number'] or ''} | "
              f"{lsn['page_from']}-{lsn['page_to']} | {counts.get(lsn['lesson_id'], 0)} | "
              f"`{lsn['lesson_id']}` |" for n, lsn in enumerate(lessons, 1)]
    if not lessons:
        lines.append("No lessons were found; the book would be indexed as one.")
    with open(target, "a", encoding="utf-8") as sink:
        sink.write("\n".join(lines) + "\n")


def _summarise(state: dict) -> None:
    """The run's outcome on the workflow's summary page, where the person who
    started it looks — not three jobs deep in a log."""
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not target:
        return
    stats = state.get("stats", {})
    extract = stats.get("extract", {})
    status = state.get("ingestion_status") or "ready"
    lines = [
        f"## {state.get('lesson_title') or state['lecture_id']}",
        "",
        f"**Status:** `{status}`",
        "",
        "| | |",
        "|---|---|",
        f"| {'Book ID' if state.get('book_id') else 'Lecture ID'} | `{state['lecture_id']}` |",
        f"| Subject ID | `{state.get('subject_id')}` |",
        f"| Pages | {extract.get('pages', 0)} |",
        f"| Unreadable pages | {extract.get('unreadable', 0)} |",
        f"| Figures | {extract.get('figures', 0)} |",
        f"| Chunks indexed | {stats.get('index', {}).get('chunks_written', 0)} |",
        f"| Topics | {stats.get('topics', {}).get('extracted', 0)} |",
        f"| Prerequisite edges | {stats.get('dependencies', {}).get('edges', 0)} |",
    ]
    results = state.get("lesson_results", [])
    if results:
        lines += ["", f"### {len(results)} lessons", "",
                  "| # | Lesson | Lesson ID | Chunks | Topics | Status |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {n} | {r['title']} | `{r['lesson_id']}` | {r['chunks']} | "
                  f"{r['topics']} | `{r['status']}` |" for n, r in enumerate(results, 1)]
    for error in state.get("errors", []):
        lines.append(f"\n> {error}")
    with open(target, "a", encoding="utf-8") as sink:
        sink.write("\n".join(lines) + "\n")


# --- files ------------------------------------------------------------------

def _dump(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as sink:
        pickle.dump(payload, sink, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path: Any) -> dict:
    with open(path, "rb") as source:
        return pickle.load(source)  # noqa: S301 — written by this workflow run only


# --- cli --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingestion.distributed",
                                     description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("plan", help="how many machines each phase gets")
    p.add_argument("source")
    p.add_argument("--read-shards", type=int, default=6)
    p.add_argument("--embed-shards", type=int, default=12)
    p.add_argument("--github-output", action="store_true",
                   help="also write the matrices to $GITHUB_OUTPUT")

    p = commands.add_parser("read", help="read one share of the pages into the cache")
    p.add_argument("source")
    add_lesson_arguments(p)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--of", type=int, required=True)
    p.add_argument("--workers", type=int, default=0,
                   help="model calls in flight (0 = ingestion.ocr.read_ahead.workers)")

    p = commands.add_parser("prepare", help="extract and chunk; write the state file")
    p.add_argument("source")
    add_lesson_arguments(p)
    p.add_argument("--out", required=True)

    p = commands.add_parser("embed", help="embed one share of the chunks")
    p.add_argument("--state", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--of", type=int, required=True)
    p.add_argument("--out", required=True)

    p = commands.add_parser("finish", help="attach vectors, then topics, dependencies, index")
    p.add_argument("--state", required=True)
    p.add_argument("--vectors", required=True, help="the folder holding every share's file")

    args = parser.parse_args(argv)
    if args.command == "plan":
        return _cmd_plan(args)

    config = get_config()
    if hasattr(args, "layout_model"):
        apply_lesson_overrides(args, config)
    setup_logging(config)
    return {
        "read": _cmd_read,
        "prepare": _cmd_prepare,
        "embed": _cmd_embed,
        "finish": _cmd_finish,
    }[args.command](args, config)


if __name__ == "__main__":
    raise SystemExit(main())
