from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class RoundConfig:
    rounds: int = 20
    seed: int = 7

@dataclass
class Step:
    round_id: int
    user_prompt: str
    adversary_op: str
    attacked_prompt: str
    defended_prompt: str
    model_output: str
    success: bool
    reward: float
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunSummary:
    total_rounds: int
    successes: int
    success_rate: float
    notes: Optional[str] = None
