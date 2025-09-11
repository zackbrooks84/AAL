from typing import List
from ..core.types import Step

def success_rate(history: List[Step]) -> float:
    if not history:
        return 0.0
    return sum(1 for s in history if s.success) / len(history)

def resilience_half_life(history: List[Step]) -> int | None:
    if not history:
        return None
    # naive half-life: find first index where trailing success rate <= half of initial
    initial = success_rate(history[: max(1, len(history)//4) ])
    target = initial / 2 if initial > 0 else 0
    for idx in range(1, len(history)+1):
        sr = success_rate(history[:idx])
        if sr <= target:
            return idx
    return None
