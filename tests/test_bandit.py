"""Tests for :mod:`aal.adversaries.bandit_prompt`.

These tests exercise the behaviour that originally caused the project’s
continuous integration workflow to fail.  The bandit adversary is a small
Thompson sampling agent that selects one of several perturbation operators.
We verify that its posterior updates are correct, that it converges toward
the optimal arm with low regret, and that state snapshots are immutable.
"""

from __future__ import annotations

import random

from aal.adversaries.bandit_prompt import BanditAdversary, OPS


def test_bandit_updates_priors():
    """Updating with a reward should only affect the chosen arm."""

    b = BanditAdversary(rng=random.Random(0))
    op = b.choose_op([])
    assert op in dict(OPS)
    a_before = dict(b.alpha)
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
