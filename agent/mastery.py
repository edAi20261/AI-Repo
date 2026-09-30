"""Mastery model — the pure rule mapping a graded event onto a topic aggregate.

No DB, no LLM (like ``retrieval.py``): the mastery rollup computes the new aggregate
via ``update`` / ``update_aggregate``, then a write-tool persists it. ``mastery_events``
is append-only; ``mastery`` is the recency-weighted rollup. A teacher correction is a
compensating event re-run through ``update`` (its ``original_score`` is kept on the event row).

Model: a graded event yields an observation ``x`` in [0,1] (``partial_score`` if
given, else 1.0/0.0 from ``correct``). The score is a difficulty-weighted EMA, so
harder items move mastery more. ``confidence`` ramps with attempts so a topic seen
< ``min_attempts`` reads as ``unknown`` rather than ``weak``. ``bloom_level`` is
recorded on the event but not yet in the numeric update (TODO: weight by bloom_level).

All numbers come from ``config.yaml mastery.*`` — nothing behavioural is hardcoded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

_DEFAULTS = {
    "alpha": 0.4,
    "difficulty_weights": {
        "correct": {"easy": 0.7, "medium": 1.0, "hard": 1.3},
        "incorrect": {"easy": 1.3, "medium": 1.0, "hard": 0.7},
    },
    "min_attempts": 2,
    "strong_threshold": 0.75,
    "weak_threshold": 0.45,
}

# Observation at/above this reads as a 'correct' outcome for choosing the weight map.
_CORRECT_CUTOFF = 0.5


@dataclass(frozen=True)
class MasteryParams:
    """Tunables lifted from config (per-call, so config stays the single source)."""
    alpha: float = 0.4                # EMA rate: weight of the newest observation vs history
    # Per-difficulty multiplier on alpha, keyed by outcome: {correct: {...}, incorrect: {...}}.
    # A hard-right / easy-wrong item moves mastery more than easy-right / hard-wrong.
    difficulty_weights: Mapping[str, Any] = None  # type: ignore[assignment]
    min_attempts: int = 2             # attempts below this -> status 'unknown' (too little evidence)
    strong_threshold: float = 0.75    # score >= this -> 'strong'
    weak_threshold: float = 0.45      # score <= this -> 'weak' (between the two -> 'developing')

    @classmethod
    def from_config(cls, config) -> "MasteryParams":
        return cls(
            alpha=config.get("mastery.alpha", _DEFAULTS["alpha"]),
            difficulty_weights=config.get(
                "mastery.difficulty_weights", _DEFAULTS["difficulty_weights"]),
            min_attempts=config.get("mastery.min_attempts", _DEFAULTS["min_attempts"]),
            strong_threshold=config.get(
                "mastery.strong_threshold", _DEFAULTS["strong_threshold"]),
            weak_threshold=config.get(
                "mastery.weak_threshold", _DEFAULTS["weak_threshold"]),
        )

    def weight_for(
        self, difficulty: Optional[str], observation: Optional[float] = None
    ) -> float:
        """Difficulty multiplier on alpha for one graded item. The outcome (from
        ``observation``, above/below ``_CORRECT_CUTOFF``) picks the correct/incorrect map,
        so hard-right and easy-wrong move mastery most. A flat legacy map (no correct/
        incorrect split) and a missing observation both fall back to a single map."""
        weights = self.difficulty_weights or _DEFAULTS["difficulty_weights"]
        diff = (difficulty or "medium").lower()
        # Legacy flat map ({easy: .., medium: .., hard: ..}) -> outcome-agnostic.
        if not isinstance(weights.get("correct"), Mapping):
            return weights.get(diff, 1.0)
        outcome = "incorrect" if (observation is not None
                                  and observation < _CORRECT_CUTOFF) else "correct"
        return (weights.get(outcome) or {}).get(diff, 1.0)


@dataclass(frozen=True)
class Mastery:
    """The ``ai.mastery`` aggregate for one (user, topic)."""
    score: float = 0.0          # current mastery estimate in [0,1] (EMA of observations)
    attempts: int = 0           # measurement occasions folded in (one assessment = one attempt)
    trend: Optional[float] = None   # signed change from the last update (+ improving / - slipping)
    confidence: float = 0.0     # ramps 0 -> 1 as attempts reach min_attempts


def observation(event: Mapping[str, Any]) -> float:
    """Map a graded event to an outcome in [0,1]: ``partial_score`` if present
    (clamped), else 1.0 for a correct answer and 0.0 otherwise."""
    # partial_score = fractional credit (0.5 = half right); else all-or-nothing from `correct`.
    ps = event.get("partial_score")
    if ps is not None:
        return _clamp(float(ps))
    return 1.0 if event.get("correct") else 0.0


def update_aggregate(
    prior: Mastery, observation: float, params: MasteryParams, *,
    difficulty_weight: float = 1.0,
) -> Mastery:
    """One EMA step from a single, already-aggregated observation in [0,1]. ``attempts``
    increments by ONE — the caller decides what a measurement occasion is (an assessment
    averages its per-topic answers into one observation, so attempts counts assessments,
    not questions). ``difficulty_weight`` scales alpha (harder items move mastery more)."""
    # Effective rate = base alpha scaled by difficulty, capped at 1 (a hard item counts more).
    alpha_eff = _clamp(params.alpha * difficulty_weight)
    # EMA: keep (1 - alpha_eff) of the old score, blend in alpha_eff of the new observation.
    new_score = (1 - alpha_eff) * prior.score + alpha_eff * _clamp(observation)
    attempts = prior.attempts + 1     # one measurement occasion
    return Mastery(
        score=new_score,
        attempts=attempts,
        trend=new_score - prior.score,                        # + improved / - slipped this step
        confidence=min(1.0, attempts / max(1, params.min_attempts)),  # 1.0 once enough attempts
    )


def update(prior: Mastery, event: Mapping[str, Any], params: MasteryParams) -> Mastery:
    """Roll a single ``event`` into ``prior`` — the one-observation convenience over
    ``update_aggregate`` (difficulty-weighted EMA, attempts +1)."""
    obs = observation(event)
    return update_aggregate(
        prior, obs, params,
        difficulty_weight=params.weight_for(event.get("difficulty"), obs))


def status(m: Mastery, params: MasteryParams) -> str:
    """Bucket a mastery aggregate: ``unknown`` while under ``min_attempts`` (so a
    thinly-sampled topic never reads as ``weak``), else by score thresholds."""
    if m.attempts < params.min_attempts:
        return "unknown"                        # too few attempts to judge yet
    if m.score >= params.strong_threshold:
        return "strong"
    if m.score <= params.weak_threshold:
        return "weak"
    return "developing"                         # between the thresholds


@dataclass(frozen=True)
class RetentionParams:
    """How long a topic stays remembered. ``R = exp(-days / S)``: stability ``S`` starts at
    ``base_stability_days`` for one assessment, grows by ``growth`` with each further one,
    scales with the score, and is capped. A topic is due for review once ``R`` falls
    below ``target``."""
    base_stability_days: float = 7.0
    growth: float = 1.5
    max_stability_days: float = 180.0
    target: float = 0.8

    @classmethod
    def from_config(cls, config) -> "RetentionParams":
        return cls(
            base_stability_days=config.get("mastery.retention.base_stability_days", 7.0),
            growth=config.get("mastery.retention.growth", 1.5),
            max_stability_days=config.get("mastery.retention.max_stability_days", 180.0),
            target=config.get("mastery.retention.target", 0.8),
        )


def stability_days(m: Mastery, params: RetentionParams) -> float:
    """Days for recall of this topic to fall to ``1/e``."""
    s = (params.base_stability_days * params.growth ** max(0, m.attempts - 1)
         * max(0.1, _clamp(m.score)))
    return min(params.max_stability_days, s)


def retention(m: Mastery, days_since: float, params: RetentionParams) -> float:
    """Estimated share of the topic still remembered ``days_since`` the last evidence."""
    return math.exp(-max(0.0, days_since) / stability_days(m, params))


def days_until_due(m: Mastery, days_since: float, params: RetentionParams) -> float:
    """Days until retention falls below the target; negative when already overdue."""
    return stability_days(m, params) * math.log(1 / params.target) - days_since


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    # Keep a value inside [0,1] so scores/weights never drift out of range.
    return max(lo, min(hi, value))
