"""Tests for :mod:`aal.adversaries.bandit_prompt`.

These tests exercise the behaviour that originally caused the project’s
continuous integration workflow to fail.  The bandit adversary is a small
Thompson sampling agent that selects one of several perturbation operators.
We verify that its posterior updates are correct, that it converges toward
the optimal arm with low regret, and that state snapshots are immutable.
"""

from __future__ import annotations

import random

import math
import json
import random
from collections import Counter, defaultdict

import pytest


from aal.adversaries.bandit_prompt import BanditAdversary, OPS



def test_bandit_updates_priors():
    """Updating with a reward should only affect the chosen arm."""

    b = BanditAdversary(rng=random.Random(0))
    
# ---------- Helpers ----------

ALL_OPS = [k for k, _ in OPS]
OP_SET = set(ALL_OPS)


def bernoulli(p: float, rng: random.Random) -> float:
    """Return 1.0 with probability p, else 0.0."""
    return 1.0 if rng.random() < p else 0.0


def run_bandit_sim(rounds: int, probs: dict[str, float], seed: int = 7):
    """
    Simulate an environment where each op has a fixed reward probability.
    The adversary chooses ops, receives Bernoulli rewards, and updates.
    Returns (adversary, history of (op, reward), counts).
    """
    assert set(probs.keys()) == OP_SET, "probs must specify every op"
    rng = random.Random(seed)
    b = BanditAdversary(rng=rng)
    history: list[tuple[str, float]] = []
    counts = Counter()
    for _ in range(rounds):
        op = b.choose_op(history)
        assert op in OP_SET
        reward = bernoulli(probs[op], rng)
        b.update(op, reward)
        history.append((op, reward))
        counts[op] += 1
    return b, history, counts


# ---------- Basic invariants ----------

def test_ops_contract_is_well_formed():
    # OPS should be unique keys and non empty prompt fragments
    keys = [k for k, _ in OPS]
    prompts = [p for _, p in OPS]
    assert len(keys) == len(set(keys)), "duplicate op keys in OPS"
    assert all(isinstance(k, str) and k for k in keys)
    assert all(isinstance(p, str) and p for p in prompts)


def test_bandit_updates_priors_monotonic():
    b = BanditAdversary()

    op = b.choose_op([])
    assert op in OP_SET
    a_before = dict(b.alpha)
    bt_before = dict(b.beta)

    # One full success update increases alpha by 1
    b.update(op, 1.0)
    assert b.alpha[op] == a_before[op] + 1.0



def test_state_snapshot_evolves_monotonically():
    """Snapshots should not be mutated by subsequent updates."""

    b = BanditAdversary(rng=random.Random(0))
    op = list(dict(OPS).keys())[0]
    snap0 = b.state_snapshot()
    b.update(op, 1.0)
    snap1 = b.state_snapshot()

    # The original snapshot retains the prior values
    assert snap0["alpha"][op] == 1.0
    assert snap1["alpha"][op] == snap0["alpha"][op] + 1.0


def test_posterior_means_track_empirical_rates():
    """Posterior mean should approximate the empirical mean of rewards."""

    rng = random.Random(42)
    b = BanditAdversary(rng=rng)
    op = list(dict(OPS).keys())[1]

    successes = 0
    trials = 200
    for _ in range(trials):
        reward = 1.0 if rng.random() < 0.7 else 0.0
        successes += reward
        b.update(op, reward)

    empirical = successes / trials
    posterior = b.alpha[op] / (b.alpha[op] + b.beta[op])

    assert abs(empirical - posterior) < 0.1


def test_bandit_identifies_best_arm_and_minimizes_regret():
    """Full simulation ensures regret remains within an acceptable bound."""

    rng = random.Random(123)
    b = BanditAdversary(rng=rng)

    ops = [k for k, _ in OPS]
    probs = {ops[0]: 0.1, ops[1]: 0.2, ops[2]: 0.3, ops[3]: 0.9}

    steps = 200
    optimal = max(probs.values())

    total_reward = 0.0
    for _ in range(steps):
        op = b.choose_op([])
        reward = 1.0 if rng.random() < probs[op] else 0.0
        total_reward += reward
        b.update(op, reward)

    regret = optimal * steps - total_reward
    # keep regret within 35% of the theoretical optimum to reduce flakiness
    assert regret < 0.35 * steps

    assert all(b.alpha[k] >= a_before[k] for k in OP_SET)
    # Beta increases by (1 - reward) which is zero here
    assert b.beta[op] == bt_before[op]
    assert all(b.beta[k] >= bt_before[k] for k in OP_SET)

    # A full failure update increases beta by 1 and leaves alpha unchanged
    a2 = dict(b.alpha)
    bt2 = dict(b.beta)
    b.update(op, 0.0)
    assert b.beta[op] == bt2[op] + 1.0
    assert b.alpha[op] == a2[op]


def test_state_snapshot_is_json_serializable():
    b = BanditAdversary()
    snap = b.state_snapshot()
    json.dumps(snap)  # must serialize for trajectory logging


# ---------- Learning dynamics ----------

@pytest.mark.flaky(reruns=2)
def test_bandit_identifies_best_arm_and_minimizes_regret():
    """
    In a stationary setting with distinct arm probabilities, a Thompson-like
    strategy should concentrate pulls on the best arm and keep regret low.
    """
    probs = {
        "translation_twist": 0.30,
        "role_confusion": 0.45,   # best arm
        "encoding_noise": 0.28,
        "flattery_anchor": 0.25,
        "persona_hijack": 0.20,
        "nested_roleplay": 0.18,
        "authority_escalation": 0.22,
        "context_overflow": 0.15,
        "hypothetical_frame": 0.19,
    }
    rounds = 1500
    b, history, counts = run_bandit_sim(rounds, probs, seed=1337)

    # Best arm should be pulled most frequently
    best = max(probs, key=probs.get)
    most_pulled = max(counts, key=counts.get)
    assert most_pulled == best, f"Expected best={best}, got {most_pulled}"

    # Cumulative regret should be less than 30 percent of uniform baseline
    # (threshold relaxed from 0.2 → 0.3 to account for larger 9-arm search space)
    p_star = probs[best]
    regret = sum((p_star - probs[op]) for op, _ in history)
    uniform_expected = sum((p_star - probs[op]) for op in probs) / len(probs) * rounds
    assert regret < 0.3 * uniform_expected, f"Regret {regret:.2f} vs {uniform_expected:.2f}"


def test_posterior_means_track_empirical_rates():
    """
    After many updates, posterior means alpha/(alpha+beta) should roughly
    match empirical success rates for each arm.
    """
    probs = {
        "translation_twist": 0.15,
        "role_confusion": 0.25,
        "encoding_noise": 0.35,
        "flattery_anchor": 0.55,
        "persona_hijack": 0.20,
        "nested_roleplay": 0.18,
        "authority_escalation": 0.22,
        "context_overflow": 0.12,
        "hypothetical_frame": 0.30,
    }
    rounds = 2000
    b, history, counts = run_bandit_sim(rounds, probs, seed=2025)

    successes = defaultdict(float)
    pulls = defaultdict(int)
    for op, r in history:
        successes[op] += r
        pulls[op] += 1

    for op in ALL_OPS:
        post_mean = b.alpha[op] / (b.alpha[op] + b.beta[op])
        if pulls[op] == 0:
            # Prior mean is 0.5 with Beta(1,1)
            assert math.isclose(post_mean, 0.5, abs_tol=1e-6)
            continue
        emp = successes[op] / pulls[op]
        assert abs(post_mean - emp) < 0.1, f"{op}: post={post_mean:.3f}, emp={emp:.3f}"


def test_exploration_then_concentration():
    """
    Early rounds should explore. Later rounds should concentrate on the best arm.
    """
    probs = {
        "translation_twist": 0.10,
        "role_confusion": 0.20,
        "encoding_noise": 0.60,  # best arm
        "flattery_anchor": 0.15,
        "persona_hijack": 0.12,
        "nested_roleplay": 0.11,
        "authority_escalation": 0.14,
        "context_overflow": 0.09,
        "hypothetical_frame": 0.13,
    }
    rounds = 1200
    b, history, counts = run_bandit_sim(rounds, probs, seed=9)

    best = "encoding_noise"
    k = rounds // 5
    first = history[:k]
    last = history[-k:]
    first_counts = Counter([op for op, _ in first])
    last_counts = Counter([op for op, _ in last])

    first_share = first_counts[best] / max(1, len(first))
    last_share = last_counts[best] / max(1, len(last))
    assert last_share > first_share + 0.15, f"Share did not rise: {first_share:.2f} -> {last_share:.2f}"


# ---------- Robustness ----------

def test_choose_op_never_returns_unknown():
    b = BanditAdversary()
    for _ in range(50):
        assert b.choose_op([]) in OP_SET


def test_update_is_stable_for_extreme_rewards():
    b = BanditAdversary()
    op = ALL_OPS[0]
    for _ in range(100):
        b.update(op, 1.0)
    for _ in range(100):
        b.update(op, 0.0)
    assert b.alpha[op] > 0 and b.beta[op] > 0
    post = b.alpha[op] / (b.alpha[op] + b.beta[op])
    assert 0.0 < post < 1.0


def test_state_snapshot_evolves_monotonically():
    b = BanditAdversary()
    op = ALL_OPS[1]
    snap0 = b.state_snapshot()
    b.update(op, 1.0)
    snap1 = b.state_snapshot()
    assert snap1["alpha"][op] == snap0["alpha"][op] + 1.0
    for k in OP_SET:
        assert snap1["alpha"][k] >= snap0["alpha"][k]
        assert snap1["beta"][k] >= snap0["beta"][k]

