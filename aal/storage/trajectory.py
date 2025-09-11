import json
from pathlib import Path
from typing import List
from ..core.types import Step

def load_trajectory(path: str) -> List[Step]:
    steps = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        steps.append(Step(**obj))
    return steps
