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
