"""Mastery model unit tests — the pure event->aggregate rule. No DB."""

from __future__ import annotations

from agent.config import get_config
from agent.mastery import Mastery, MasteryParams, observation, status, update


def _params(**over) -> MasteryParams:
    base = MasteryParams(
        alpha=0.3,
        difficulty_weights={"easy": 0.7, "medium": 1.0, "hard": 1.3},
        min_attempts=3,
        strong_threshold=0.75,
        weak_threshold=0.45,
    )
    return MasteryParams(**{**base.__dict__, **over})


# --- observation ------------------------------------------------------------

def test_observation_partial_over_correct():
    assert observation({"correct": True}) == 1.0
    assert observation({"correct": False}) == 0.0
    assert observation({"partial_score": 0.6, "correct": False}) == 0.6  # partial wins
    assert observation({"partial_score": 1.5}) == 1.0                    # clamped


# --- update / EMA -----------------------------------------------------------

def test_update_moves_toward_observation_and_counts_attempts():
    p = _params()
    m = update(Mastery(), {"correct": True, "difficulty": "medium"}, p)
    assert 0 < m.score <= 1
    assert m.attempts == 1
    assert m.trend == m.score            # from 0 -> new
    # a wrong answer next pulls it back down
    m2 = update(m, {"correct": False, "difficulty": "medium"}, p)
    assert m2.score < m.score
    assert m2.trend < 0                  # getting worse
    assert m2.attempts == 2


def test_difficulty_weight_hard_moves_more_than_easy():
    p = _params()
    easy = update(Mastery(), {"correct": True, "difficulty": "easy"}, p)
    hard = update(Mastery(), {"correct": True, "difficulty": "hard"}, p)
    assert hard.score > easy.score       # same correct answer, harder moves more


def test_asymmetric_difficulty_weight_by_outcome():
    # Uses the real config (nested correct/incorrect maps).
    p = MasteryParams.from_config(get_config())
    assert p.weight_for("hard", 1.0) > p.weight_for("easy", 1.0)   # correct: harder counts more
    assert p.weight_for("easy", 0.0) > p.weight_for("hard", 0.0)   # wrong: easy miss counts more
    # An easy question answered WRONG pulls mastery down harder than a hard one answered wrong.
    easy_wrong = update(Mastery(score=0.8, attempts=3),
                        {"correct": False, "difficulty": "easy"}, p)
    hard_wrong = update(Mastery(score=0.8, attempts=3),
                        {"correct": False, "difficulty": "hard"}, p)
    assert easy_wrong.score < hard_wrong.score


def test_legacy_flat_difficulty_map_still_supported():
    # A flat {easy,medium,hard} map (no correct/incorrect split) stays outcome-agnostic.
    p = _params(difficulty_weights={"easy": 0.7, "medium": 1.0, "hard": 1.3})
    assert p.weight_for("hard", 0.0) == p.weight_for("hard", 1.0) == 1.3


def test_confidence_ramps_to_one_at_min_attempts():
    p = _params(min_attempts=3)
    m = Mastery()
    confidences = []
    for _ in range(3):
        m = update(m, {"correct": True, "difficulty": "medium"}, p)
        confidences.append(m.confidence)
    assert confidences[0] < 1 and confidences[1] < 1
    assert confidences[2] == 1.0


# --- status buckets ---------------------------------------------------------

def test_status_unknown_below_min_attempts():
    p = _params(min_attempts=3)
    assert status(Mastery(score=0.9, attempts=2), p) == "unknown"  # thin sample


def test_status_thresholds_when_confident():
    p = _params(min_attempts=3, strong_threshold=0.75, weak_threshold=0.45)
    assert status(Mastery(score=0.8, attempts=5), p) == "strong"
    assert status(Mastery(score=0.4, attempts=5), p) == "weak"
    assert status(Mastery(score=0.6, attempts=5), p) == "developing"


# --- correction path --------------------------------------------------------

def test_correction_event_lowers_topic():
    p = _params()
    m = update(Mastery(), {"correct": True, "difficulty": "medium"}, p)
    # teacher correction: the answer was actually wrong -> compensating event
    corrected = update(m, {"correct": False, "original_score": 1.0,
                           "difficulty": "medium"}, p)
    assert corrected.score < m.score
    assert corrected.attempts == m.attempts + 1


# --- config wiring ----------------------------------------------------------

def test_params_from_config_reads_yaml():
    p = MasteryParams.from_config(get_config())
    assert 0 < p.alpha <= 1
    assert p.weight_for("hard") >= p.weight_for("easy")
    assert p.min_attempts >= 1
