"""Tests for new AAL features: extended operators, ELO, defense_delta, dashboard."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aal.adversaries.bandit_prompt import BanditAdversary, OPS
from aal.core.loop import Loop
from aal.core.types import Step
from aal.defenders.coach_self_reflection import SelfReflectDefender
from aal.metrics.basic import defense_delta, learning_curve, success_rate
from aal.metrics.elo import compute_elo, per_op_stats, DEFAULT_RATING
from aal.models.dummy import DummyModel
from aal.safety.policy import simple_scorer
from aal.dashboard import render


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_history(successes: list[bool]) -> list[Step]:
    return [
        Step(
            round_id=i + 1,
            user_prompt="test",
            adversary_op=f"op_{i % 3}",
            attacked_prompt="",
            defended_prompt="",
            model_output="BREACH" if s else "SAFE",
            success=s,
            reward=1.0 if s else 0.0,
        )
        for i, s in enumerate(successes)
    ]


# ---------------------------------------------------------------------------
# extended operator tests
# ---------------------------------------------------------------------------

def test_ops_count():
    assert len(OPS) == 9, f"Expected 9 ops, got {len(OPS)}"


def test_all_new_ops_have_non_empty_prefix():
    new_op_keys = {
        "persona_hijack", "nested_roleplay", "authority_escalation",
        "context_overflow", "hypothetical_frame",
    }
    op_dict = dict(OPS)
    for key in new_op_keys:
        assert key in op_dict, f"Missing op: {key}"
        assert len(op_dict[key]) > 10, f"Prefix too short for {key}"


def test_bandit_explores_new_ops():
    rng_seed = 42
    import random
    adversary = BanditAdversary(rng=random.Random(rng_seed))
    history = []
    chosen = set()
    for _ in range(50):
        op = adversary.choose_op(history)
        chosen.add(op)
        history.append(
            Step(0, "", op, "", "", "", False, 0.0)
        )
        adversary.update(op, 0.0)
    # all 9 ops should be explored eventually
    assert len(chosen) == 9


def test_bandit_apply_includes_new_ops():
    adversary = BanditAdversary()
    for key, _ in OPS:
        result = adversary.apply(key, "test prompt")
        assert "test prompt" in result
        assert "1234" in result


# ---------------------------------------------------------------------------
# ELO tests
# ---------------------------------------------------------------------------

def test_elo_initial_all_equal():
    history = _make_history([False] * 10)
    # All op_{i%3} — use same op throughout to simplify
    h = [Step(i + 1, "", "op_a", "", "", "", False, 0.0) for i in range(10)]
    elos = compute_elo(h)
    assert "op_a" in elos
    assert "defender" in elos


def test_elo_adversary_rises_on_breach():
    # 10 consecutive breaches — adversary ELO should rise above default
    h = [Step(i + 1, "", "op_a", "", "", "", True, 1.0) for i in range(10)]
    elos = compute_elo(h)
    assert elos["op_a"] > DEFAULT_RATING
    assert elos["defender"] < DEFAULT_RATING


def test_elo_defender_rises_on_blocks():
    # 10 consecutive blocks — defender ELO should rise
    h = [Step(i + 1, "", "op_a", "", "", "", False, 0.0) for i in range(10)]
    elos = compute_elo(h)
    assert elos["defender"] > DEFAULT_RATING
    assert elos["op_a"] < DEFAULT_RATING


def test_per_op_stats_structure():
    history = _make_history([True, False, True, False, True])
    stats = per_op_stats(history)
    for op, s in stats.items():
        assert "attempts" in s
        assert "successes" in s
        assert "rate" in s
        assert "elo" in s
        assert 0.0 <= s["rate"] <= 1.0


def test_per_op_stats_rates():
    h = [
        Step(1, "", "op_a", "", "", "", True, 1.0),
        Step(2, "", "op_a", "", "", "", True, 1.0),
        Step(3, "", "op_b", "", "", "", False, 0.0),
    ]
    stats = per_op_stats(h)
    assert stats["op_a"]["rate"] == pytest.approx(1.0)
    assert stats["op_b"]["rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# defense_delta tests
# ---------------------------------------------------------------------------

def test_defense_delta_improving():
    # First half: all breaches.  Second half: no breaches.
    history = _make_history([True] * 10 + [False] * 10)
    dd = defense_delta(history)
    assert dd < 0, "Expected negative delta (defender improved)"


def test_defense_delta_degrading():
    history = _make_history([False] * 10 + [True] * 10)
    dd = defense_delta(history)
    assert dd > 0, "Expected positive delta (defender degraded)"


def test_defense_delta_stable():
    history = _make_history([True, False] * 10)
    dd = defense_delta(history)
    assert dd == pytest.approx(0.0, abs=0.2)


def test_defense_delta_short_history():
    history = _make_history([True])
    assert defense_delta(history) == 0.0


# ---------------------------------------------------------------------------
# learning_curve tests
# ---------------------------------------------------------------------------

def test_learning_curve_length():
    history = _make_history([True, False] * 10)
    curve = learning_curve(history)
    assert len(curve) == len(history)


def test_learning_curve_values_in_range():
    history = _make_history([True, False, True, False, True])
    curve = learning_curve(history)
    for v in curve:
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# full loop with new operators
# ---------------------------------------------------------------------------

def test_loop_with_all_ops():
    adversary = BanditAdversary()
    defender = SelfReflectDefender()
    model = DummyModel()
    loop = Loop(adversary, defender, model, simple_scorer)
    with tempfile.TemporaryDirectory() as d:
        summary = loop.run(rounds=18, out_dir=d)  # 9 ops × 2 warm-up rounds
        assert summary.total_rounds == 18
        traj = Path(d) / "trajectory.jsonl"
        assert traj.exists()


# ---------------------------------------------------------------------------
# dashboard tests
# ---------------------------------------------------------------------------

def test_dashboard_renders():
    adversary = BanditAdversary()
    defender = SelfReflectDefender()
    model = DummyModel()
    loop = Loop(adversary, defender, model, simple_scorer)
    with tempfile.TemporaryDirectory() as d:
        loop.run(rounds=20, out_dir=d)
        from aal.storage.trajectory import load_trajectory
        history = load_trajectory(str(Path(d) / "trajectory.jsonl"))
        output = render(history, run_name="test-run")
        assert "AAL Dashboard" in output
        assert "SUMMARY" in output
        assert "OPERATOR BREAKDOWN" in output


def test_dashboard_empty():
    output = render([])
    assert "No data" in output
