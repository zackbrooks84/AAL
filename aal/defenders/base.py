from typing import Dict, Any, List

class DefenderBase:
    def wrap(self, attacked_prompt: str, history: List[Any]) -> str:
        raise NotImplementedError

    def update(self, breached: bool, model_output: str) -> None:
        pass

    def state_snapshot(self) -> Dict[str, Any]:
        return {}
