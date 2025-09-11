from typing import List, Dict, Any
import random

class AdversaryBase:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose_op(self, history: List[Any]) -> str:
        raise NotImplementedError

    def apply(self, op: str, prompt: str) -> str:
        raise NotImplementedError

    def update(self, op: str, reward: float) -> None:
        pass

    def state_snapshot(self) -> Dict[str, Any]:
        return {}
