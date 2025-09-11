# tests/test_metrics_half_life.py
#
# Purpose
# - Validate AAL's learning-under-attack metric: "resilience half-life".
# - Show that when breach rate drops after a defense update, the half-life:
#     1) exists (finite) in realistic scenarios,
#     2) falls within a reasonable window after the change point, and
#     3) is earlier when the improvement is stronger (monotonicity).
#
# Notes
# - This is a pure-metric test; no models or I/O are needed.
# - We construct synthetic Step histories with a clear change point.
# - The baseline half-life implementation is in aal/metrics/basic.py.

from __future__ import annotations

from typing import List
import pytest

from aal.core.types import Step
from aal.metrics.basic import success_rate, resilience_half_life


def _mk_step(round_id: int, success: bool) -> Step:
    # Minimal Step; only fields used by metrics are "success" (bool) and list length
    return Step(
        round_id=round_id,
        user_prompt="",
        adversary_op="op",
        attacked_prompt="",
        defended_prompt="",
        model_output="",
        success=success,
        reward=1.0 if success else 0.0,
        meta={},
    )


def _history_with_piecewise_rate(
    n1: int, rate1: float, n2: int, rate2: float
) -> List[Step]:
    """
    Build a two-phase history:
      - First n1 rounds with success fraction ~= rate1
      - Then n2 rounds with success fraction ~= rate2
    We deterministically distribute successes via periodic pattern so the rate is exact.
    """
    hist: List[Step] = []

    def add_block(start_idx: int, n: int, rate: float):
        # Success every k rounds where k = int(1/rate) if rate>0; else no success
        if rate <= 0.0:
            for i in range(n):
                hist.append(_mk_step(start_idx + i + 1, False))
            return
        # Create exactly round(rate * n) successes distributed evenly
        total_succ = round(rate * n)
        succ_positions = set()
        if total_succ > 0:
            spacing = n / float(total_succ)
            # pick indices 0..n-1 closest to i*spacing
            for i in range(total_succ):
                pos = int(round(i * spacing))
                pos = min(max(pos, 0), n - 1)
                succ_positions.add(pos)
        for i in range(n):
            hist.append(_mk_step(start_idx + i + 1, i in succ_positions))

    add_block(0, n1, rate1)
    add_block(n1, n2, rate2)
    return hist


def test_success_rate_sanity_checks():
    # 10 rounds with 3 successes
    hist = [_mk_step(i + 1, i < 3) for i in range(10)]
    assert success_rate(hist) == 0.3

    # Empty history -> 0.0 by design
    assert success_rate([]) == 0.0


def test_half_life_none_when_rate_constant():
    # Breach rate constant at 0.5 the whole time
    hist = _history_with_piecewise_rate(n1=40, rate1=0.5, n2=60, rate2=0.5)
    hl = resilience_half_life(hist)
    # With no improvement, success rate never drops to half of the initial 0.5 -> target 0.25
    assert hl is None


def test_half_life_exists_and_is_within_window_after_change():
    """
    Scenario:
      - Early phase: high breach rate (0.80) for 40 rounds.
      - Later phase: improved defense, breach rate 0.20 for 80 rounds.
    Expect:
      - A finite half-life exists.
      - It occurs after the change point, and not too late.
    """
    n1, r1, n2, r2 = 40, 0.80, 80, 0.20
    hist = _history_with_piecewise_rate(n1=n1, rate1=r1, n2=n2, rate2=r2)
    hl = resilience_half_life(hist)
    assert hl is not None, "Half-life should exist when breach rate improves"

    # The implementation defines initial as rate over the first quarter of the whole run.
    # With N=120, first quarter is 30 rounds, initial ≈ 0.80, target = 0.40.
    # After the change at round 41, the cumulative rate will trend downward.
    # We do not require an exact index, just that it falls reasonably soon after the change.
    assert 41 <= hl <= 70, f"Half-life {hl} should be soon after change point"


def test_half_life_monotonicity_under_stronger_improvement():
    """
    Two scenarios with identical early phase (0.75 for 40 rounds).
    After round 41:
      - Weak improvement: 0.30 breach rate.
      - Strong improvement: 0.10 breach rate.
    Expect:
      - Strong improvement yields an earlier (smaller) half-life index.
    """
    n1, r1, n2 = 40, 0.75, 80

    weak = _history_with_piecewise_rate(n1=n1, rate1=r1, n2=n2, rate2=0.30)
    strong = _history_with_piecewise_rate(n1=n1, rate1=r1, n2=n2, rate2=0.10)

    hl_weak = resilience_half_life(weak)
    hl_strong = resilience_half_life(strong)

    assert hl_weak is not None and hl_strong is not None
    assert hl_strong < hl_weak, (
        "Expected earlier half-life with stronger improvement, "
        f"got {hl_strong=} vs {hl_weak=}"
    )
