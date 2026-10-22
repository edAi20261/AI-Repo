"""Load a running AI service with parallel requests and report when each began.

The client's target is a first reply within 3 s with 50 chats in parallel. A
streamed turn answers its headers at once, so "started" is measured at the
first server-sent event, not at the response.

Turns carry no session_id, so nothing is remembered and nothing is written.

    python -m scripts.load_chat --url http://localhost:8000 --user <student id> \
        --subject <subject id> --n 10
    python -m scripts.load_chat ... --n 50 --suggest 20 --summary 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from shared.config import Config, get_config
from system.security import SERVICE_TOKEN_HEADER

_MESSAGES = (
    "Explain this lesson's main idea with an example.",
    "What should I remember from this lesson?",
    "Give me a short summary of the key steps.",
)


@dataclass
class Timing:
    first_event: Optional[float] = None
    first_progress: Optional[float] = None
    first_stage: Optional[float] = None
    total: Optional[float] = None
    error: Optional[str] = None
    stages: list[str] = field(default_factory=list)


async def _turn(client: httpx.AsyncClient, url: str, body: dict) -> Timing:
    timing = Timing()
    start = time.perf_counter()
    try:
        async with client.stream("POST", f"{url}/ai/chat/stream", json=body) as response:
            response.raise_for_status()
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line[len("event: "):]
                    elapsed = time.perf_counter() - start
                    timing.first_event = timing.first_event or elapsed
                    if event == "progress" and timing.first_progress is None:
                        timing.first_progress = elapsed
                elif line.startswith("data: ") and event == "progress":
                    stage = json.loads(line[len("data: "):]).get("stage")
                    timing.stages.append(stage)
                    if stage != "queued" and timing.first_stage is None:
                        timing.first_stage = time.perf_counter() - start
                elif line.startswith("data: ") and event == "error":
                    timing.error = line[len("data: "):]
    except httpx.HTTPError as exc:
        timing.error = repr(exc)
    timing.total = time.perf_counter() - start
    return timing


async def _call(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> Timing:
    timing = Timing()
    start = time.perf_counter()
    try:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        timing.error = repr(exc)
    timing.total = time.perf_counter() - start
    return timing


def _report(name: str, timings: list[Timing], measure: str) -> None:
    values = sorted(getattr(t, measure) for t in timings if getattr(t, measure) is not None)
    failed = [t.error for t in timings if t.error]
    if not values:
        print(f"{name:<22} no {measure} recorded; {len(failed)} failed")
        return
    p95 = values[min(len(values) - 1, round(0.95 * (len(values) - 1)))]
    print(f"{name:<22} n={len(timings):<3} median {statistics.median(values):6.2f}s  "
          f"p95 {p95:6.2f}s  max {values[-1]:6.2f}s  failed {len(failed)}")
    for error in failed[:3]:
        print(f"    {error[:200]}")


async def _run(args: argparse.Namespace) -> None:
    get_config()
    headers = {SERVICE_TOKEN_HEADER: Config.secret("AI_WEBHOOK_SHARED_SECRET") or ""}
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(headers=headers, limits=limits, timeout=timeout) as client:
        turns = [
            _turn(client, args.url, {
                "user_id": args.user, "user_type": "student", "subject_id": args.subject,
                "message": _MESSAGES[i % len(_MESSAGES)], "language": args.language,
            })
            for i in range(args.n)
        ]
        suggest = [
            _call(client, "POST", f"{args.url}/ai/topics/suggest", json={
                "subject_id": args.subject,
                "questions": [{"text": _MESSAGES[i % len(_MESSAGES)]}
                              for i in range(args.suggest)],
            })
        ] if args.suggest else []
        summaries = [
            _call(client, "GET", f"{args.url}/ai/learners/{args.user}/summary",
                  params={"subject_ids": [args.subject]})
            for _ in range(args.summary)
        ]
        results = await asyncio.gather(*turns, *suggest, *summaries)
    turned = results[:len(turns)]
    suggested = results[len(turns):len(turns) + len(suggest)]
    summarised = results[len(turns) + len(suggest):]
    if turned:
        _report("chat first event", turned, "first_event")
        _report("chat first progress", turned, "first_progress")
        _report("chat first stage done", turned, "first_stage")
        _report("chat total", turned, "total")
    if suggested:
        _report(f"suggest {args.suggest} questions", suggested, "total")
    if summarised:
        _report("learner summary", summarised, "total")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--user", required=True, help="a student's user id")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--n", type=int, default=10, help="parallel chat turns")
    parser.add_argument("--suggest", type=int, default=0,
                        help="questions in one topic-suggestion call sent alongside")
    parser.add_argument("--summary", type=int, default=0,
                        help="parallel learner-summary calls sent alongside")
    parser.add_argument("--language", default="en")
    parser.add_argument("--timeout", type=float, default=180.0)
    asyncio.run(_run(parser.parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
