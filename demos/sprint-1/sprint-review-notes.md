# Sprint 1 — technical notes for review

The demo write-up is in [README.md](README.md).

## Models & infrastructure

| Role | Model / component | Where it runs | API cost |
|---|---|---|---|
| Page transcription (scanned / maths pages) + figure captions | **Qwen3-VL-235B-A22B-Instruct** (OpenRouter), fallback Claude Sonnet | OpenRouter API | yes |
| Topic extraction | **Moonshot Kimi-K2.5** (OpenRouter) | OpenRouter API | yes |
| Prerequisite (dependency) derivation | **Moonshot Kimi-K2.5** (OpenRouter) | OpenRouter API | yes |
| Embeddings (bilingual index) | **BGE-M3** (BAAI/bge-m3), 1024-dim, normalised | **local, CPU** | none |
| Stores | PostgreSQL (`ai.*` tables) + Qdrant (vector index) | — | — |

Born-digital pages use PyMuPDF text directly (no vision call); only scanned/maths
pages and figure captions hit the vision model.

## Measured per-lecture time

From a re-ingest of a 9-page **digital** lecture (17 figures, 9 chunks, 6 topics):
**total ≈ 12m 48s**, broken down by stage:

| Stage | Time | Note |
|---|---|---|
| extract | ~1m 25s | 9 digital pages + 17 figure caption calls (vision) |
| chunk | seconds | local |
| embed | ~40s | BGE-M3 local (incl. ~9s model load) |
| topics | ~1m 35s | 1 LLM call |
| **dependencies** | **~9m 04s** | **1 LLM call per topic, sequential — the bottleneck** |
| index | ~4s | DB + Qdrant writes |

The initial 5-lecture mixed set averaged **~15–20 min/lecture**; Arabic scanned
lectures run slower because every page is a vision call (digital pages are not).

**Planning average: ~15 min/lecture** (range ~13–25 depending on page count and
scanned-vs-digital).

## Full-corpus scaling estimate

Corpus: **39 subject PDFs × 10–15 lectures each ≈ 390–585 lectures.**

**Sequential (one lecture at a time):**

| Lectures | @ 15 min | @ 20 min |
|---|---|---|
| 390 | ~4 days | ~5.5 days |
| 585 | ~6 days | ~8 days |

→ **~4–8 days of continuous processing** on the current single-threaded path.

**Parallel (K lectures at once)** — lectures are independent, so wall-clock ≈
sequential ÷ K:

| K (concurrent) | Approx wall-clock |
|---|---|
| 5 | ~1–1.5 days |
| 8 | ~0.5–1 day |

**Constraints on parallelism:**
- Bounded by **OpenRouter rate limits** (requests- and tokens-per-minute) — beyond
  the tier's cap, requests throttle/retry and the speedup flattens.
- Total **API cost is unchanged** by parallelism (same number of calls) — only
  wall-clock improves.
- Needs a small amount of orchestration (worker pool / queue) — not in place yet.

## Cost

Cost is driven by LLM call volume per lecture (current models, via OpenRouter):
- **Vision:** ~1 call per *scanned* page + 1 per figure caption. (Digital pages = 0.)
- **Topics:** ~1–2 calls. **Dependencies:** ~1 call per topic.
- Example lecture above ≈ **~24 calls** (17 captions + topics + 6 dependency calls),
  0 page-vision (it was digital). A scanned Arabic lecture adds ~1 vision call/page.

**Cost is measured automatically.** The pipeline captures the per-call cost OpenRouter
returns (both vision and text calls) and reports it per lecture as `cost` (`cost_usd`,
`prompt_tokens`, `completion_tokens`, `calls`). **Measured on the 5-lesson sample** (metered
re-ingest, 2026-08-07):

| Lecture (subject) | Cost | LLM calls | Notes |
|---|---|---|---|
| Social Studies AR | $0.095 | 18 | digital but very text-heavy (72k input tokens) |
| Science EN | $0.068 | 24 | digital, 17 figures |
| Social Studies AR — 2 | $0.075 | 17 | mostly digital |
| Math EN | $0.078 | 21 | maths (vision route), 8 figures |
| Science FR | $0.097 | 41 | **scanned** (10 vision page-calls) + 24 figures |
| **Total (5 lectures)** | **$0.413** | — | |

| Metric | Value |
|---|---|
| **$/lecture (measured average)** | **≈ $0.083** |
| **Full corpus (× 390–585)** | **≈ $32 – $48** |

Per-lecture range **$0.068–$0.097**. Cost tracks two things: **vision calls** (scanned/maths
pages + figure captions) and **text volume** fed to topic/dependency extraction (tokens). The
two priciest were the token-heavy Arabic history lecture (~72k input tokens) and the scanned,
figure-heavy French one. So the whole-corpus AI cost is a **one-time ~$30–50**, not a recurring
concern. Parallelism doesn't change this total — only the wall-clock.

**Still needed:** the key's **rate-limit tier** (RPM/TPM / credit-based
concurrency) for the parallelism estimate above, and billing ownership for the bulk backfill.

## Recommendations

1. **Run the full backfill in Sprint 4 on AWS**, as a **queued worker-pool job**, not
   a one-shot script — the pipeline is per-lecture independent and already
   queue/CLI-triggerable.
2. **Biggest single quick win: parallelise the dependency calls** (one per topic)
   *within* a lecture — that stage alone was ~70% of this lecture's time. Same idea
   for the per-page vision and per-figure caption calls. This could cut per-lecture
   time from ~13 min toward ~3–4 min before any cross-lecture parallelism.
3. **Then parallelise across lectures** (K workers), tuned to the OpenRouter rate tier.
4. In production the **publish-triggered ingest** spreads lectures out as teachers
   publish — only the **one-time backfill of existing content** is the heavy job.

## Idempotency evidence (re-ingest)

Re-ingesting one lecture leaves the stores unchanged — no duplication or orphaning:

| Check | Before | After |
|---|---|---|
| total `ai.topics` | 32 | 32 |
| dependency edges | 45 | 45 |
| total `ai.figures` | 63 | 63 |
| lecture topic ids | 6 | same 6 (reused) |
| lecture figure keys | 17 | same 17 |

Re-ingest stats for the lecture: `topics: {extracted: 6, reused: 6, new: 0}`.
