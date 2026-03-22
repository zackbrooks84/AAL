"""Basic metrics used throughout AAL."""

from collections import deque
from typing import List

from ..core.types import Step


def success_rate(history: List[Step]) -> float:
    """Return the fraction of successful steps in ``history``.

    Parameters
    ----------
    history:
        Sequence of :class:`~aal.core.types.Step` objects.

    Returns
    -------
    float
        ``0.0`` for empty histories, otherwise successes divided by length.
    """

    if not history:
        return 0.0
    return sum(1 for s in history if s.success) / len(history)


def resilience_half_life(history: List[Step]) -> int | None:
    """Return first round index where breach rate halves.

    The initial breach rate is measured over the first quarter of the run.
    The metric then scans forward using a sliding window of the same length
    and returns the earliest round where the trailing success rate falls to
    half of the initial value.  ``None`` is returned if the threshold is not
    reached.

    Parameters
    ----------
    history:
        Sequence of :class:`~aal.core.types.Step` objects.

    Returns
    -------
    int | None
        1-indexed round number where the rate halves, or ``None`` if it never
        does.
    """

    n = len(history)
    if n == 0:
        return None

    window = max(1, n // 4)
    successes = sum(1 for s in history[:window] if s.success)
    initial = successes / window
    target = initial / 2 if initial > 0 else 0.0

    if successes / window <= target:
        return window

    dq: deque[Step] = deque(history[:window])
    for idx in range(window, n):
        dq.append(history[idx])
        if history[idx].success:
            successes += 1
        old = dq.popleft()
        if old.success:
            successes -= 1

        if successes / window <= target:
            return idx + 1  # convert to 1-indexed round number

    return None


def defense_delta(history: List[Step], halves: int = 2) -> float:
    """Measure how much the defender improved across the run.

    Splits the history into equal halves and computes the change in
    adversary success rate from the first half to the last half.  A
    *negative* delta means the defender improved (fewer breaches later);
    a *positive* delta means it got worse.

    Parameters
    ----------
    history:
        Ordered sequence of :class:`~aal.core.types.Step` objects.
    halves:
        Number of segments to split the history into.  The delta is
        measured between the first and last segment.

    Returns
    -------
    float
        ``last_half_rate - first_half_rate``.  ``0.0`` for histories
        too short to split.
    """
    n = len(history)
    if n < halves * 2:
        return 0.0

    seg = n // halves
    first = history[:seg]
    last = history[n - seg:]

    first_rate = sum(1 for s in first if s.success) / seg
    last_rate = sum(1 for s in last if s.success) / seg
    return last_rate - first_rate


def learning_curve(history: List[Step], window: int = 5) -> List[float]:
    """Rolling success rate across the run.

    Parameters
    ----------
    history:
        Ordered sequence of :class:`~aal.core.types.Step` objects.
    window:
        Rolling window size.

    Returns
    -------
    list of float
        One rate per step (using a trailing window).
    """
    rates = []
    for i, _ in enumerate(history):
        start = max(0, i - window + 1)
        chunk = history[start : i + 1]
        rates.append(sum(1 for s in chunk if s.success) / len(chunk))
    return rates
