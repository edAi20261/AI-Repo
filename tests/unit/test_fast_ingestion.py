"""Reading a book once, on every core, and from several machines.

A 306-page scanned textbook was read twice per page — once for its layout, once
for its words — in two passes, each rendering every page again on the one thread
that owns the document: 0.83 s a page at 300 dpi on a 2-vCPU machine, twice per
page. A re-run of a book already read rendered all of it again to learn that
nothing needed reading.

And embedding — 2.5 s a chunk on 2 vCPUs, twenty-five minutes for a 600-chunk
book — had no way to use a second machine.
"""

from __future__ import annotations

import importlib
import threading

import pytest

from ingestion.layout.blocks import Block, PageMeta, ParsedPage
from ingestion.schemas import PageTranscription

extract = importlib.import_module("ingestion.stages.extract")
read_ahead_module = importlib.import_module("ingestion.read_ahead")

LECTURE = "11111111-1111-1111-1111-111111111111"
PARAGRAPH = ("The moon pulls the sea towards it, and the water on the far side of the "
             "Earth is left behind, so most coasts see two high tides and two low tides "
             "every day. When the sun and the moon line up, their pulls add together and "
             "the tides are larger; these are called spring tides. When they pull at right "
             "angles, the tides are smaller, and these are called neap tides.")


# --- doubles -------------------------------------------------------------------

class MemoryCache:
    """``PageCache`` with a dict for a table: the same keys, the same contract."""

    def __init__(self, shape):
        self.shape = shape
        self.rows: dict[tuple[str, int], object] = {}
        self.enabled = True
        self.puts = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def get(self, lecture_id, page):
        found = self.get_many(lecture_id, [page])
        return found.get(page)

    def get_many(self, lecture_id, pages):
        found = {p: self.rows[(lecture_id, p)] for p in pages if (lecture_id, p) in self.rows}
        self.hits += len(found)
        self.misses += len(set(pages)) - len(found)
        return found

    def put(self, lecture_id, page, parsed, **_):
        assert isinstance(parsed, self.shape), "a page stored under the wrong shape"
        with self._lock:
            self.rows[(lecture_id, page)] = parsed
            self.puts += 1

    def report(self):
        return {"cache_hits": self.hits, "cache_misses": self.misses,
                "cache_enabled": self.enabled}


class Calls:
    def __init__(self):
        self.renders = 0
        self.layout = 0
        self.words = 0
        self._lock = threading.Lock()

    def bump(self, name):
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)


class Vision:
    is_stub = False

    def __init__(self, calls, *, figures=False):
        self.calls = calls
        self.figures = figures

    def transcribe_page(self, png, *, language_hint=None, math=False):
        self.calls.bump("words")
        boxes = [{"bbox": (0.1, 0.1, 0.5, 0.5)}] if self.figures else []
        return PageTranscription(text="Tides rise and fall twice a day.", figures=boxes)

    def caption_figure(self, image, *, language_hint=None):
        return None


@pytest.fixture
def scanned_book(tmp_path):
    """A PDF of pages with an image and no text layer, as a scanned book is."""
    import pymupdf

    from PIL import Image

    picture = tmp_path / "scan.png"
    Image.new("RGB", (60, 80), (200, 190, 180)).save(picture)

    def make(pages: int) -> str:
        doc = pymupdf.open()
        for _ in range(pages):
            page = doc.new_page(width=300, height=400)
            page.insert_image(page.rect, filename=str(picture))
        path = tmp_path / f"book-{pages}.pdf"
        doc.save(path)
        doc.close()
        return str(path)

    return make


@pytest.fixture
def wired(monkeypatch):
    """The extract stage and the read-ahead, with caches, models and renders counted."""
    from shared.config import Config

    calls = Calls()
    layout_cache = MemoryCache(ParsedPage)
    word_caches = {False: MemoryCache(PageTranscription), True: MemoryCache(PageTranscription)}

    real_render = extract.pdf.render_page_png

    def render(page, dpi=300):
        calls.bump("renders")
        return real_render(page, dpi=36)

    def parse(png, meta, **_):
        calls.bump("layout")
        return ParsedPage(meta=meta.model_copy(update={"source_tier": 3}), blocks=[
            Block(text=f"Tides, page {meta.page}. " + PARAGRAPH,
                  bbox=(30, 120, 270, 300), source_tier=3)])

    monkeypatch.setattr(extract.pdf, "render_page_png", render)
    monkeypatch.setattr(extract.tier3, "parse_page", parse)
    monkeypatch.setattr(extract, "_page_cache", lambda cfg, layout_cfg: layout_cache)
    monkeypatch.setattr(extract, "_transcription_cache", lambda cfg, math: word_caches[math])

    config = Config.load()
    ctx = type("Ctx", (), {"config": config, "vision": Vision(calls)})()
    return type("Wired", (), {"calls": calls, "ctx": ctx, "config": config,
                              "layout_cache": layout_cache, "word_caches": word_caches})


def _state(**extra):
    return {"lecture_id": LECTURE, "language_hint": "en", "subject_is_math": False,
            "stats": {}, "errors": [], **extra}


# --- reading ahead ---------------------------------------------------------------

def test_a_page_is_rendered_once_for_both_of_its_model_calls(scanned_book, wired):
    """Each pass rendered the page again: two renders per page, on one thread."""
    book = scanned_book(5)

    report = read_ahead_module.read_ahead(book, _state(), wired.ctx, processes=1, workers=4)

    assert wired.calls.renders == 5
    assert wired.calls.layout == 5
    assert wired.calls.words == 5
    assert report["layout_read"] == 5 and report["transcribed"] == 5
    assert len(wired.layout_cache.rows) == 5
    assert len(wired.word_caches[False].rows) == 5


def test_after_reading_ahead_the_extract_stage_renders_nothing_and_buys_nothing(
        scanned_book, wired):
    """The read-ahead is only worth anything if the stage after it finds every
    page under the keys it looks up. A run that found every page cached still
    rendered all of them, twice, to learn that nothing needed reading."""
    book = scanned_book(6)
    read_ahead_module.read_ahead(book, _state(), wired.ctx, processes=1, workers=3)
    before = (wired.calls.renders, wired.calls.layout, wired.calls.words)

    doc = extract.pdf.open_document(book)
    try:
        counts = {"figures": 0}
        layout = extract._prefetch_layout_reads(
            doc, _state(), layout_cfg=wired.config.ingestion["layout"],
            cache=wired.layout_cache, min_chars=40, render_dpi=300,
            config=wired.config, workers=4, counts=counts)
        words = extract._prefetch_transcriptions(
            doc, _state(), wired.ctx, min_chars=40, render_dpi=300, is_math=False,
            garbage_cfg={}, default_lang="en", workers=4, counts=counts)
    finally:
        doc.close()

    assert (wired.calls.renders, wired.calls.layout, wired.calls.words) == before
    assert all(parsed is not None for _, _, _, parsed, _ in layout.values())
    assert counts["layout_read_by_vision"] == 6
    assert {no: result.text for no, (result, _, _) in words.items()} == dict.fromkeys(
        range(1, 7), "Tides rise and fall twice a day.")


def test_a_cached_page_with_figures_is_still_rendered_to_cut_them(scanned_book, wired):
    """The transcription is cached and the figure crops are not. A page whose
    figures were skipped because its words were cached would lose them."""
    book = scanned_book(2)
    wired.ctx.vision = Vision(wired.calls, figures=True)
    read_ahead_module.read_ahead(book, _state(), wired.ctx, processes=1, workers=2)
    renders = wired.calls.renders

    cut = []
    original = extract._figures_from_boxes

    def figures(state, ctx, png, page_no, boxes, lang, counts=None):
        assert png, "a figure cut from no image"
        cut.append(page_no)
        return []

    extract._figures_from_boxes = figures
    try:
        doc = extract.pdf.open_document(book)
        try:
            extract._prefetch_transcriptions(
                doc, _state(), wired.ctx, min_chars=40, render_dpi=300, is_math=False,
                garbage_cfg={}, default_lang="en", workers=2, counts={"figures": 0})
        finally:
            doc.close()
    finally:
        extract._figures_from_boxes = original

    assert sorted(cut) == [1, 2]
    assert wired.calls.renders == renders + 2
    assert wired.calls.words == 2, "a page already read was paid for again"


def test_a_book_read_by_several_machines_is_read_exactly_once(scanned_book, wired):
    book = scanned_book(23)
    for shard in range(4):
        pages = read_ahead_module.pages_of_shard(23, shard, 4)
        read_ahead_module.read_ahead(book, _state(), wired.ctx, pages=pages,
                                     processes=1, workers=3)

    assert wired.calls.words == 23
    assert wired.calls.layout == 23
    assert sorted(p for _, p in wired.word_caches[False].rows) == list(range(1, 24))


def test_the_shares_of_a_book_cover_every_page_once():
    shares = [read_ahead_module.pages_of_shard(306, i, 6) for i in range(6)]
    every = [p for share in shares for p in share]
    assert sorted(every) == list(range(1, 307))
    assert max(map(len, shares)) - min(map(len, shares)) <= 1
    with pytest.raises(ValueError):
        read_ahead_module.pages_of_shard(306, 6, 6)


def test_a_page_outside_the_book_is_refused_not_skipped(scanned_book, wired):
    with pytest.raises(ValueError):
        read_ahead_module.read_ahead(scanned_book(3), _state(), wired.ctx, pages=[4],
                                     processes=1)


def test_reading_ahead_with_no_vision_key_refuses_rather_than_caching_stub_output(
        scanned_book, wired):
    """The stub reader returns placeholder text. Cached, it would be served to
    every later run as the page's words."""
    wired.ctx.vision.is_stub = True
    with pytest.raises(read_ahead_module.ReadAheadUnavailable):
        read_ahead_module.read_ahead(scanned_book(2), _state(), wired.ctx, processes=1)
    assert wired.calls.words == 0


def test_reading_ahead_into_a_cache_nobody_can_reach_is_refused(scanned_book, wired):
    """Every page read would be thrown away and bought again by the pipeline."""
    wired.layout_cache.enabled = False
    with pytest.raises(read_ahead_module.ReadAheadUnavailable):
        read_ahead_module.read_ahead(scanned_book(2), _state(), wired.ctx, processes=1)
    assert wired.calls.words == 0 and wired.calls.layout == 0


def test_a_page_that_fails_costs_that_page_not_the_book(scanned_book, wired, caplog):
    import logging

    def flaky(png, *, language_hint=None, math=False):
        raise RuntimeError("connection reset")

    wired.ctx.vision.transcribe_page = flaky
    with caplog.at_level(logging.ERROR):
        report = read_ahead_module.read_ahead(scanned_book(3), _state(), wired.ctx,
                                              processes=1, workers=2)

    assert report["failed"] == 3
    assert report["layout_read"] == 3
    assert any("could not be read ahead" in r.message for r in caplog.records)


def test_the_render_pool_draws_the_same_pixels_as_the_pipeline(scanned_book):
    """Two processes rendering one book must produce the image a single thread
    would — the cache keys say nothing about who drew the page."""
    from ingestion.utils import pdf
    from ingestion.utils.render_pool import PageRenderer

    book = scanned_book(3)
    doc = pdf.open_document(book)
    try:
        expected = [pdf.render_page_png(doc[i], dpi=50) for i in range(3)]
    finally:
        doc.close()

    with PageRenderer(book, dpi=50, processes=2) as renderer:
        got = [renderer.render(n) for n in (1, 2, 3)]

    assert got == expected


# --- the page cache, a book at a time ---------------------------------------------

def test_a_whole_book_is_looked_up_in_one_round_trip(fake_postgres):
    from ingestion.layout.page_cache import PageCache
    from shared.config import Config

    meta = PageMeta(doc_id="d", page=4, width=300, height=400)
    stored = ParsedPage(meta=meta, blocks=[Block(text="Tides", bbox=(0, 0, 10, 10))])
    db = fake_postgres(rows=[(4, stored.model_dump(mode="json")),
                             (5, {"text": "a transcription", "figures": []})])

    cache = PageCache(Config.load(), model="m", prompt_version="v")
    found = cache.get_many(LECTURE, [4, 5, 6])

    assert list(found) == [4]
    assert found[4].blocks[0].text == "Tides"
    assert cache.report() == {"cache_hits": 1, "cache_misses": 2, "cache_enabled": True}
    assert len(db.statements) == 1


def test_an_unreachable_cache_answers_a_book_lookup_with_nothing(fake_postgres):
    from ingestion.layout.page_cache import PageCache
    from shared.config import Config

    fake_postgres(explode=True)
    cache = PageCache(Config.load(), model="m", prompt_version="v")

    assert cache.get_many(LECTURE, [1, 2]) == {}
    assert cache.enabled is False
    assert cache.report()["cache_misses"] == 0


# --- embedding on several machines -------------------------------------------------

def _chunks(n):
    from ingestion.state import Chunk

    return [Chunk(lecture_id=LECTURE, text=f"The moon pulls the sea, part {i}.",
                  heading_path="Tides", page_from=i + 1, page_to=i + 1, language="en",
                  order_index=i) for i in range(n)]


def _hash_config():
    from shared.config import Config

    config = Config.load()
    config._data["embeddings"]["provider"] = "hash"
    return config


def test_a_book_embedded_on_several_machines_is_indexed_as_on_one():
    """Shares are attached by position. One off by one and every chunk after it
    is searched by its neighbour's vector, which nothing would ever notice."""
    from ingestion.distributed import FORMAT, collect, shard_bounds, texts_digest
    from shared.embeddings import get_embedder

    embed = importlib.import_module("ingestion.stages.embed")

    config = _hash_config()
    embedder = get_embedder(config)
    alone = embed.run({"chunks": _chunks(17), "stats": {}},
                      type("Ctx", (), {"config": config, "embedder": embedder})())

    texts = embed.embedding_texts(_chunks(17), config)
    prepared = {"texts": texts, "digest": texts_digest(texts)}
    shares = []
    for shard in reversed(range(5)):
        lo, hi = shard_bounds(len(texts), shard, 5)
        shares.append({"format": FORMAT, "digest": prepared["digest"], "lo": lo, "hi": hi,
                       "dense": embedder.embed(texts[lo:hi]),
                       "sparse": embedder.embed_sparse(texts[lo:hi]),
                       "dimension": embedder.dimension,
                       "model": config.get("embeddings.model")})

    dense, sparse, dimension = collect(prepared, shares, model=config.get("embeddings.model"),
                                       hybrid=True)
    split = embed.attach({"chunks": _chunks(17), "stats": {}}, config, dense, sparse,
                         dimension=dimension)

    assert [c.embedding for c in split["chunks"]] == [c.embedding for c in alone["chunks"]]
    assert [c.sparse for c in split["chunks"]] == [c.sparse for c in alone["chunks"]]
    assert split["stats"]["embed"] == alone["stats"]["embed"]


def _shares(texts, digest, cuts, model="BAAI/bge-m3"):
    from ingestion.distributed import FORMAT

    return [{"format": FORMAT, "digest": digest, "lo": lo, "hi": hi,
             "dense": [[0.0]] * (hi - lo), "sparse": [{}] * (hi - lo),
             "dimension": 1, "model": model} for lo, hi in cuts]


@pytest.mark.parametrize("cuts, problem", [
    ([(0, 4), (5, 10)], "have no vectors"),
    ([(0, 6), (5, 10)], "embedded twice"),
    ([(0, 4)], "have no vectors"),
])
def test_vectors_that_leave_a_chunk_out_or_twice_are_refused(cuts, problem):
    from ingestion.distributed import collect, texts_digest

    texts = [f"t{i}" for i in range(10)]
    prepared = {"texts": texts, "digest": texts_digest(texts)}
    with pytest.raises(ValueError, match=problem):
        collect(prepared, _shares(texts, prepared["digest"], cuts),
                model="BAAI/bge-m3", hybrid=True)


def test_vectors_made_from_other_chunks_are_refused():
    """A share from another run of the same book — chunked before a config
    change — has the right count and the wrong text."""
    from ingestion.distributed import collect, texts_digest

    texts = [f"t{i}" for i in range(4)]
    prepared = {"texts": texts, "digest": texts_digest(texts)}
    stale = _shares(texts, texts_digest(["other"] * 4), [(0, 4)])
    with pytest.raises(ValueError, match="different chunks"):
        collect(prepared, stale, model="BAAI/bge-m3", hybrid=True)


def test_vectors_from_another_model_are_refused():
    from ingestion.distributed import collect, texts_digest

    texts = ["a", "b"]
    prepared = {"texts": texts, "digest": texts_digest(texts)}
    with pytest.raises(ValueError, match="this run embeds with"):
        collect(prepared, _shares(texts, prepared["digest"], [(0, 2)], model="other"),
                model="BAAI/bge-m3", hybrid=True)


def test_the_shares_of_the_chunks_tile_them_exactly():
    from ingestion.distributed import shard_bounds

    for total in (0, 1, 7, 600):
        for of in (1, 5, 12):
            bounds = [shard_bounds(total, i, of) for i in range(of)]
            assert bounds[0][0] == 0 and bounds[-1][1] == total
            assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))


def test_a_small_book_is_not_split_across_more_readers_than_it_can_use():
    from ingestion.distributed import plan

    assert len(plan(25, readers=6, embedders=12)["read"]["include"]) == 3
    assert len(plan(306, readers=6, embedders=12)["read"]["include"]) == 6
    assert len(plan(306, readers=6, embedders=12)["embed"]["include"]) == 12


def test_stages_cannot_be_run_out_of_order():
    """``topics`` before ``embed`` tags chunks that have no vectors and reports
    success."""
    from ingestion.graph import run_stages

    with pytest.raises(ValueError):
        run_stages({}, None, ("topics", "embed"))


# --- the whole run, split across machines ------------------------------------------

def test_a_book_run_across_machines_ends_in_the_index_one_machine_would_build(
        scanned_book, wired, monkeypatch, tmp_path):
    """Every command the workflow runs, in the order it runs them, each building
    its own context from scratch as a fresh machine would — against the same
    book run through the ordinary pipeline. The index must not be able to tell
    which one built it."""
    import ingestion.context as context
    import ingestion.distributed as distributed
    import ingestion.lesson as lesson
    from ingestion import new_state, run_ingestion
    from ingestion.context import PipelineContext

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost")
    monkeypatch.setenv("QDRANT_URL", "dryrun")
    for key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GOOGLE_API_KEY", "MOONSHOT_API_KEY", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOCAL_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    config = wired.config
    config._data["embeddings"]["provider"] = "hash"
    monkeypatch.setattr(distributed, "get_config", lambda: config)
    monkeypatch.setattr(context, "get_vision_client", lambda cfg: wired.ctx.vision)

    class _Backend:
        def get_subject(self, subject_id):
            return {"name": "English", "language": "en"}

    monkeypatch.setattr(lesson, "get_backend_client", lambda cfg, **k: _Backend())

    book = scanned_book(11)
    lesson_args = ["--lecture-id", LECTURE, "--subject-id", "s1",
                   "--lesson-title", "Tides and the Moon"]

    for shard in range(3):
        assert distributed.main(["read", book, *lesson_args,
                                 "--shard", str(shard), "--of", "3", "--workers", "2"]) == 0
    calls_after_reading = (wired.calls.layout, wired.calls.words)

    state_file = tmp_path / "state" / "state.pkl"
    assert distributed.main(["prepare", book, *lesson_args, "--out", str(state_file)]) == 0
    assert (wired.calls.layout, wired.calls.words) == calls_after_reading, \
        "prepare paid for a page the readers had already read"

    for shard in range(4):
        assert distributed.main(["embed", "--state", str(state_file), "--shard", str(shard),
                                 "--of", "4", "--out",
                                 str(tmp_path / "vectors" / f"vectors-{shard}.pkl")]) == 0

    finished = {}
    monkeypatch.setattr(distributed, "report", lambda lecture_id, state: finished.update(state))
    assert distributed.main(["finish", "--state", str(state_file),
                             "--vectors", str(tmp_path / "vectors")]) == 0

    alone = run_ingestion(
        new_state(lecture_id=LECTURE, subject_id="s1", source_uri=book,
                  subject_name="English", lesson_title="Tides and the Moon",
                  language_hint="en"),
        ctx=PipelineContext.build(config), config=config)

    def index(state):
        return [(c.text, c.page_from, c.embedding, c.sparse, c.topic_ids)
                for c in state["chunks"]]

    assert len(finished["chunks"]) >= 11, "a page went missing between machines"
    assert index(finished) == index(alone)
    assert finished["stats"]["embed"] == alone["stats"]["embed"]
    assert finished["stats"]["index"] == alone["stats"]["index"]
    assert finished["ingestion_status"] == alone["ingestion_status"] == "ready"


def test_a_book_is_split_into_its_lessons_and_each_is_indexed_under_its_own_id(
        scanned_book, wired, monkeypatch, tmp_path):
    import ingestion.context as context
    import ingestion.distributed as distributed
    import ingestion.lesson as lesson
    from ingestion.books import lessons_from
    from ingestion.schemas import LessonBreakdown, LessonItem, UnitItem

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost")
    monkeypatch.setenv("QDRANT_URL", "dryrun")
    for key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GOOGLE_API_KEY", "MOONSHOT_API_KEY", "S3_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOCAL_ARTIFACT_DIR", str(tmp_path / "artifacts"))

    config = wired.config
    config._data["embeddings"]["provider"] = "hash"
    monkeypatch.setattr(distributed, "get_config", lambda: config)
    monkeypatch.setattr(context, "get_vision_client", lambda cfg: wired.ctx.vision)
    subject = "8c2f0a10-0001-4a00-9000-00000000000a"
    monkeypatch.setattr(lesson, "_subject_row", lambda cfg, sid: {
        "subject_id": sid, "title": None, "name": "English", "language": "en",
        "grade_level": "Grade 7"})

    def split(state, cfg):
        breakdown = LessonBreakdown(units=[UnitItem(title="Unit 1", unit_number=1, lessons=[
            LessonItem(title="Lesson 1", start_page=1),
            LessonItem(title="Lesson 2", start_page=6)])])
        return lessons_from(breakdown, book_id=state["lecture_id"], first_page=1,
                            last_page=11)

    monkeypatch.setattr(distributed, "lessons_for", split)

    class _Catalog:
        registered: list = []
        settled: list = []

        def __init__(self, cfg):
            pass

        def register(self, *, subject_id, book_title, lessons, namespace="syllabus"):
            _Catalog.registered.append((subject_id, [lsn["title"] for lsn in lessons]))

        def finished(self, lesson_id, *, status, errors, version):
            _Catalog.settled.append((lesson_id, status))

        def stale(self, book_id, kept):
            return []

        def archive(self, lesson_ids):
            pass

    monkeypatch.setattr(distributed, "LessonCatalog", _Catalog)

    book = scanned_book(11)
    args = ["--subject-id", subject]
    assert distributed.main(["read", book, *args, "--shard", "0", "--of", "1",
                             "--workers", "2"]) == 0
    state_file = tmp_path / "state" / "state.pkl"
    assert distributed.main(["prepare", book, *args, "--out", str(state_file)]) == 0
    for shard in range(2):
        assert distributed.main(["embed", "--state", str(state_file), "--shard", str(shard),
                                 "--of", "2", "--out",
                                 str(tmp_path / "vectors" / f"vectors-{shard}.pkl")]) == 0
    finished = {}
    monkeypatch.setattr(distributed, "report", lambda lecture_id, state: finished.update(state))
    assert distributed.main(["finish", "--state", str(state_file),
                             "--vectors", str(tmp_path / "vectors")]) == 0

    lessons = finished["lessons"]
    assert _Catalog.registered == [(subject, ["Lesson 1", "Lesson 2"])]
    assert _Catalog.settled == [(lsn["lesson_id"], "ready") for lsn in lessons]
    for chunk in finished["chunks"]:
        owner = lessons[0] if chunk.page_from < 6 else lessons[1]
        assert chunk.lecture_id == owner["lesson_id"]
        assert chunk.chunk_id
    results = finished["lesson_results"]
    assert [r["lesson_id"] for r in results] == [lsn["lesson_id"] for lsn in lessons]
    assert all(r["chunks"] for r in results)
    assert sum(r["chunks"] for r in results) == len(finished["chunks"])
    assert finished["book_id"] not in {lsn["lesson_id"] for lsn in lessons}


# --- a book with nothing typed about it -----------------------------------------------

def _lesson_args(*argv):
    import argparse

    from ingestion.lesson import add_lesson_arguments, fill_in_identity

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    add_lesson_arguments(parser)
    args = parser.parse_args(list(argv))
    fill_in_identity(args)
    return args


def test_every_machine_gives_an_untyped_book_the_same_identity(scanned_book):
    """Each machine of a split run works the ids out for itself. A random id
    would make six machines read six different books into the cache, and
    the run that builds the index would find none of them."""
    book = scanned_book(3)
    first, second = _lesson_args(book), _lesson_args(book)
    assert first.lecture_id == second.lecture_id
    assert first.subject_id == second.subject_id


def test_the_same_book_uploaded_again_replaces_itself(scanned_book, tmp_path):
    """The index replaces a lecture by its id. An id that changed between two
    uploads of one book would index it twice, and every answer would cite it
    twice."""
    import shutil

    book = scanned_book(3)
    copy = tmp_path / "renamed.pdf"
    shutil.copyfile(book, copy)
    assert _lesson_args(book).lecture_id == _lesson_args(str(copy)).lecture_id
    assert _lesson_args(scanned_book(4)).lecture_id != _lesson_args(book).lecture_id


def test_books_named_under_one_subject_share_it(scanned_book):
    """Prerequisites are drawn from the subject's topic catalogue, so two
    English books that landed in two subjects could never link to each other."""
    a = _lesson_args(scanned_book(3), "--subject-name", "English")
    b = _lesson_args(scanned_book(4), "--subject-name", " english ")
    assert a.subject_id == b.subject_id
    assert a.lecture_id != b.lecture_id


def test_ids_that_were_typed_are_kept(scanned_book):
    args = _lesson_args(scanned_book(2), "--lecture-id", "L1", "--subject-id", "S1",
                        "--lesson-title", "Tides")
    assert (args.lecture_id, args.subject_id, args.lesson_title) == ("L1", "S1", "Tides")


def _lesson_state(*argv):
    import argparse

    from ingestion.lesson import add_lesson_arguments, lesson_state
    from shared.config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    add_lesson_arguments(parser)
    return lesson_state(parser.parse_args(list(argv)), get_config())


def test_a_book_takes_its_subject_title_and_language_from_its_lesson(
        scanned_book, fake_postgres):
    db = fake_postgres(fetchone=[("8c2f0a10-0001-4a00-9000-00000000000a", "Gem — English", "English", "en", "Grade 7")])
    state = _lesson_state(scanned_book(2), "--lecture-id", "9d3f1b20-0003-4a00-9000-000000000007")
    assert db.last_args == ("9d3f1b20-0003-4a00-9000-000000000007",)
    assert (state["lecture_id"], state["subject_id"]) == ("9d3f1b20-0003-4a00-9000-000000000007", "8c2f0a10-0001-4a00-9000-00000000000a")
    assert state["lesson_title"] == "Gem — English"
    assert (state["subject_name"], state["language_hint"]) == ("English", "en")


def test_a_book_under_a_subject_that_does_not_exist_is_refused_before_any_page_is_read(
        scanned_book, fake_postgres):
    from ingestion.lesson import SubjectNotFound

    fake_postgres(fetchone=[None])
    with pytest.raises(SubjectNotFound):
        _lesson_state(scanned_book(2), "--subject-id", "8c2f0a10-0001-4a00-9000-00000000000a")


def test_looking_up_a_subject_opens_one_connection_not_a_pool(
        scanned_book, fake_postgres, monkeypatch):
    from shared import postgres

    subject = "8c2f0a10-0001-4a00-9000-00000000000a"
    fake_postgres(fetchone=[(subject, None, "English", "en", "Grade 7")])

    def no_pool(*args, **kwargs):
        raise AssertionError("a one-off lookup opened a connection pool")

    monkeypatch.setattr(postgres, "_pool_for", no_pool)
    assert _lesson_state(scanned_book(2), "--subject-id", subject)["subject_id"] == subject


def test_a_book_takes_its_grade_and_language_from_its_subject(scanned_book, fake_postgres):
    subject = "8c2f0a10-0001-4a00-9000-00000000000a"
    fake_postgres(fetchone=[(subject, None, "English", "en", "Grade 7")])
    state = _lesson_state(scanned_book(2), "--subject-id", subject)
    assert state["subject_id"] == subject
    assert state["book_id"] == state["lecture_id"]
    assert (state["subject_name"], state["language_hint"]) == ("English", "en")


def test_a_lesson_lookup_nobody_can_reach_falls_back_to_the_typed_ids(
        scanned_book, fake_postgres):
    fake_postgres(explode=True)
    state = _lesson_state(scanned_book(2), "--lecture-id", "9d3f1b20-0003-4a00-9000-000000000007", "--subject-id", "S1")
    assert (state["lecture_id"], state["subject_id"]) == ("9d3f1b20-0003-4a00-9000-000000000007", "S1")


def test_a_lesson_id_that_is_not_an_id_is_refused(scanned_book, fake_postgres):
    from ingestion.lesson import LessonNotFound

    fake_postgres()
    with pytest.raises(LessonNotFound):
        _lesson_state(scanned_book(2), "--lecture-id", "the gem book")


def test_an_untitled_book_is_named_after_the_file_it_came_from(scanned_book):
    """A Drive download is saved as book.pdf; its real name travels beside it.
    Drive's right-to-left names carry invisible direction marks, which must
    not reach a citation a student reads."""
    from pathlib import Path

    book = Path(scanned_book(2))
    fetched = book.with_name("book.pdf")
    book.rename(fetched)
    Path(str(fetched) + ".name").write_text(
        "‎⁨جيم_انجليزي_1ع_الترم_الاول⁩.pdf", encoding="utf-8")

    assert _lesson_args(str(fetched)).lesson_title == "جيم انجليزي 1ع الترم الاول"


def test_a_drive_file_name_in_arabic_arrives_readable():
    """Drive sends the name as raw UTF-8; read as the header's Latin-1 it is
    accented gibberish."""
    from scripts.fetch_book import _filename

    raw = 'attachment; filename="كتاب.pdf"'.encode("utf-8").decode("latin-1")
    assert _filename(raw) == "كتاب.pdf"


def test_a_split_run_given_only_the_book_reads_each_page_once(
        scanned_book, wired, monkeypatch, tmp_path):
    """The readers and prepare each derive the book's id on their own. If they
    disagreed, prepare would look the pages up under another lecture, find
    none, and pay to read the whole book a second time."""
    import pickle

    import ingestion.context as context
    import ingestion.distributed as distributed
    import ingestion.lesson as lesson

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost")
    monkeypatch.setenv("QDRANT_URL", "dryrun")
    monkeypatch.setenv("LOCAL_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setattr(distributed, "get_config", lambda: wired.config)
    monkeypatch.setattr(context, "get_vision_client", lambda cfg: wired.ctx.vision)
    monkeypatch.setattr(lesson, "get_backend_client",
                        lambda cfg, **k: (_ for _ in ()).throw(ValueError("no backend")))

    book = scanned_book(7)
    for shard in range(2):
        assert distributed.main(["read", book,
                                 "--shard", str(shard), "--of", "2", "--workers", "2"]) == 0
    paid = (wired.calls.layout, wired.calls.words)
    assert paid == (7, 7)

    out = tmp_path / "state.pkl"
    assert distributed.main(["prepare", book, "--out", str(out)]) == 0
    assert (wired.calls.layout, wired.calls.words) == paid

    with open(out, "rb") as source:
        state = pickle.load(source)["state"]
    assert state["lecture_id"] == _lesson_args(book).lecture_id
    assert state["chunks"]


# --- a book's language, from its own words --------------------------------------------

ENGLISH = ("Read the email and answer the questions. My best friend lives in another "
           "city, so we write to each other every week about school and our projects. ")
ARABIC = ("اقرأ البريد الإلكتروني وأجب عن الأسئلة. يعيش صديقي المفضل في مدينة أخرى، "
          "لذلك نكتب لبعضنا كل أسبوع عن المدرسة ومشاريعنا. ")


def test_an_arabic_book_is_read_as_arabic():
    from ingestion.stages.extract import book_language

    assert book_language(ARABIC * 10, ("ar", "en", "fr", "de")) == "ar"


def test_an_english_workbook_glossed_in_arabic_stays_english():
    """Egyptian English textbooks translate every word list into Arabic. At the
    per-message rule (30% Arabic letters makes a turn Arabic) the whole book
    would be indexed and answered as Arabic."""
    from ingestion.stages.extract import book_language
    from shared.language.script import arabic_ratio

    glossed = (ENGLISH + ARABIC) * 6
    assert 0.3 <= arabic_ratio(glossed) < 0.5
    assert book_language(glossed, ("ar", "en", "fr", "de")) == "en"


def test_too_little_text_is_not_a_language():
    """A guess looks exactly like a decision, and is worse than the default."""
    from ingestion.stages.extract import book_language

    assert book_language("Unit 1", ("ar", "en")) is None


def test_nobody_is_asked_for_the_language_of_a_book_already_read(scanned_book, wired):
    """The readers have already paid for every page's words; the language is
    in them. Chunks are labelled with it, so an Arabic book left on the
    English default is answered in the wrong language."""
    book = scanned_book(4)
    wired.ctx.vision.transcribe_page = (
        lambda png, *, language_hint=None, math=False: PageTranscription(text=ARABIC * 3))
    read_ahead_module.read_ahead(book, _state(language_hint=None), wired.ctx,
                                 processes=1, workers=2)

    state = _state(language_hint=None)
    doc = extract.pdf.open_document(book)
    try:
        found = extract._language_of_book(doc, state, wired.config, ("ar", "en", "fr", "de"),
                                          min_chars=40, garbage_cfg={})
    finally:
        doc.close()
    assert found == "ar"


# --- the first run of the ingest workflow ---------------------------------------------

def test_a_secret_pasted_with_a_space_around_it_still_works(monkeypatch):
    """QDRANT_URL was saved in the repository's secrets as " https://…". Every
    machine of the first ingest run failed its preflight with "Host ' https'
    contains invalid character ' '" before reading a page."""
    from shared.config import Config

    monkeypatch.setenv("QDRANT_URL", " https://example.qdrant.io\n")
    assert Config.secret("QDRANT_URL") == "https://example.qdrant.io"
    monkeypatch.delenv("QDRANT_URL")
    assert Config.secret("QDRANT_URL", "dryrun") == "dryrun"


def test_the_storage_preflight_writes_and_clears_a_probe(monkeypatch, tmp_path):
    """The check handed the uploader its key as the bytes and its bytes as the
    key, so it failed with "a bytes-like object is required" on every machine
    — whatever the S3 credentials were."""
    from scripts import preflight_ingest
    from shared.config import Config

    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_ARTIFACT_DIR", str(tmp_path))
    result = preflight_ingest._storage(Config.load())

    assert "wrote and removed" in result
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_a_file_named_book_is_not_titled_book(scanned_book):
    """A book fetched from a link whose own name is book.pdf was cited to
    students as "book"."""
    from pathlib import Path

    book = Path(scanned_book(2))
    fetched = book.with_name("book.pdf")
    book.rename(fetched)
    Path(str(fetched) + ".name").write_text("book.pdf", encoding="utf-8")
    assert _lesson_args(str(fetched)).lesson_title == "Untitled book"


def test_aws_keys_saved_with_a_newline_still_sign_requests(monkeypatch):
    """AWS_ACCESS_KEY_ID was saved in the repository's secrets with a newline at
    its end. boto3 read it from the environment itself, past Config.secret, and
    put it in the signature header as "Credential=AKIA…\\n/…", which the HTTP
    client refused: the second ingest run failed its storage check on every
    machine."""
    from shared import storage

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE\n")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", " secret\n")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    credentials = storage._s3_client("eu-central-1", None)._request_signer._credentials

    assert credentials.access_key == "AKIAEXAMPLE"
    assert credentials.secret_key == "secret"


def test_without_keys_in_the_environment_boto3_keeps_its_own_chain(monkeypatch):
    """The service on ECS has no keys in its environment: it signs with its task
    role, found by boto3's own chain, which explicit empty keys would break."""
    from shared import storage

    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    assert storage._aws_keys() == {}


@pytest.mark.parametrize("pasted", [
    'AWS_SECRET_ACCESS_KEY=k7Ej/abc+def',
    'AWS_SECRET_ACCESS_KEY     k7Ej/abc+def\n',
    '"k7Ej/abc+def"',
    "'k7Ej/abc+def'",
    ' k7Ej/abc+def\n',
])
def test_a_secret_pasted_with_its_name_or_quotes_is_read_as_its_value(monkeypatch, pasted):
    """The third ingest run failed every machine with S3's SignatureDoesNotMatch:
    the access key was right and the secret beside it was not what was typed
    into the form. A line copied whole from .env or from a table carries the
    name with it, and nothing downstream can tell."""
    from shared.config import Config

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", pasted)
    assert Config.secret("AWS_SECRET_ACCESS_KEY") == "k7Ej/abc+def"


def test_a_value_that_merely_starts_like_its_name_is_left_alone(monkeypatch):
    from shared.config import Config

    monkeypatch.setenv("S3_BUCKET", "S3_BUCKETS-archive")
    assert Config.secret("S3_BUCKET") == "S3_BUCKETS-archive"
