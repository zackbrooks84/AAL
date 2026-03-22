"""ELO rating system for adversary operators and defenders.

Each adversary operator and the defender are rated as players in a
zero-sum game.  When an operator breaches the defender the operator wins
(ELO goes up, defender goes down) and vice-versa.
"""
from __future__ import annotations

from typing import Dict, List

from ..core.types import Step

DEFAULT_RATING = 1200.0
K_FACTOR = 32.0


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _update(rating: float, expected: float, actual: float, k: float = K_FACTOR) -> float:
    return rating + k * (actual - expected)


def compute_elo(
    history: List[Step],
    k: float = K_FACTOR,
) -> Dict[str, float]:
    """Compute ELO ratings for every adversary operator and the defender.

    Parameters
    ----------
    history:
        Ordered sequence of :class:`~aal.core.types.Step` objects from a run.
    k:
        ELO K-factor.  Higher values make ratings more volatile.

    Returns
    -------
    dict
        Mapping of ``{"op_name": rating, ..., "defender": rating}``.
        All ratings start at :data:`DEFAULT_RATING`.
    """
    # Collect all operator names seen
    ops_seen: set[str] = {s.adversary_op for s in history}
    ratings: Dict[str, float] = {op: DEFAULT_RATING for op in ops_seen}
    ratings["defender"] = DEFAULT_RATING

    for step in history:
        op = step.adversary_op
        r_op = ratings[op]
        r_def = ratings["defender"]

        # adversary wins (breach=1), defender wins (breach=0)
        adversary_score = 1.0 if step.success else 0.0
        defender_score = 1.0 - adversary_score

        e_op = _expected(r_op, r_def)
        e_def = _expected(r_def, r_op)

        ratings[op] = _update(r_op, e_op, adversary_score, k)
        ratings["defender"] = _update(r_def, e_def, defender_score, k)

    return ratings


def per_op_stats(history: List[Step]) -> Dict[str, Dict[str, float]]:
    """Per-operator success rate, attempt count, and ELO.

    Parameters
    ----------
    history:
        Ordered sequence of :class:`~aal.core.types.Step` objects.

    Returns
    -------
    dict
        ``{op_name: {"attempts": n, "successes": n, "rate": float, "elo": float}}``
    """
    counts: Dict[str, Dict[str, int]] = {}
    for step in history:
        op = step.adversary_op
        if op not in counts:
            counts[op] = {"attempts": 0, "successes": 0}
        counts[op]["attempts"] += 1
        if step.success:
            counts[op]["successes"] += 1

    elos = compute_elo(history)

    return {
        op: {
            "attempts": v["attempts"],
            "successes": v["successes"],
            "rate": v["successes"] / v["attempts"] if v["attempts"] else 0.0,
            "elo": round(elos.get(op, DEFAULT_RATING), 1),
        }
        for op, v in counts.items()
    }
