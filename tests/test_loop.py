from aal.core.loop import Loop
from aal.adversaries.bandit_prompt import BanditAdversary
from aal.defenders.coach_self_reflection import SelfReflectDefender
from aal.models.dummy import DummyModel
from aal.safety.policy import simple_scorer
import tempfile, json, pathlib

def test_loop_runs_and_writes_files():
    adversary = BanditAdversary()
    defender = SelfReflectDefender()
    model = DummyModel()
    loop = Loop(adversary, defender, model, simple_scorer)
    with tempfile.TemporaryDirectory() as d:
        summary = loop.run(rounds=6, out_dir=d)
        assert summary.total_rounds == 6
        tpath = pathlib.Path(d) / "trajectory.jsonl"
        assert tpath.exists()
        lines = tpath.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 6
        step = json.loads(lines[0])
        assert "adversary_op" in step and "model_output" in step
