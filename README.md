# Adaptive Adversarial Looping (AAL)

AAL is a research-grade evaluation harness that pits an adaptive adversary against a learning defender across repeated rounds. 
Instead of one-shot red teaming, AAL measures how models and defenses learn under attack and how quickly vulnerabilities decay.

**Why it is new**
- Adversary adapts based on the last defense and its own regret signal.
- Defender can update within the loop to reduce breach success across rounds.
- Metrics focus on *learning under pressure*: resilience half-life, defense delta, and regret-conditioned ELO for adversaries and defenses.
- Produces a public trajectory dataset of attack-defense rounds for reproducible science.

> Safe by design: default tasks are harmless simulators that never request real harm. You can plug in your own tasks and safety policies later if needed.

## Quick start

AAL supports Python 3.10–3.13. Continuous integration tests run on Ubuntu and Windows natively (no Git Bash required).

**Windows (cmd):**
```cmd
python -m venv .venv && .venv\Scripts\activate.bat
pip install -U pip pytest
pip install -e .
aal run --rounds 20 --adversary bandit --defender self_reflect --model dummy --out ./runs/run1
aal report --in ./runs/run1
pytest -q
```

**Windows (PowerShell):**
```powershell
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -U pip pytest
pip install -e .
aal run --rounds 20 --adversary bandit --defender self_reflect --model dummy --out ./runs/run1
aal report --in ./runs/run1
pytest -q
```

**Linux / macOS:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip pytest
pip install -e .
aal run --rounds 20 --adversary bandit --defender self_reflect --model dummy --out ./runs/run1
aal report --in ./runs/run1
pytest -q
```

You can also run via `python -m aal.cli` without installing. Pass `--config examples/configs/quick.yaml` to load parameters from a file (CLI args override config values).

## Concepts

- **Loop**: repeated rounds where adversary proposes an attack, defender wraps a user task prompt, model responds, and a scorer assesses success.
- **Adaptive adversary**: multi-armed bandit chooses attack operators based on rewards.
- **Learning defender**: configurable strategies such as self-reflection that update prompts based on previous failures.
- **Metrics**: success rate over time, resilience half-life, defense delta, ELO rating for both sides.

## Repository layout

```
aal/
  core/loop.py            - main tournament engine
  core/types.py           - dataclasses for steps and configs
  adversaries/            - adversary policies (bandit baseline included)
  defenders/              - defender policies (self-reflection baseline included)
  models/                 - provider interfaces and dummy model for tests
  metrics/                - success metrics, ELO, learning curves
  logging_utils/          - event logging and pretty printing
  storage/                - JSONL trajectory writer and schema
  safety/                 - simple content policy stubs
  cli.py                  - Typer-like CLI implemented with argparse
examples/
  configs/quick.yaml      - example config
tests/
  test_loop.py
  test_bandit.py
docs/
  design.md
  metrics.md
.github/workflows/ci.yaml
pyproject.toml
LICENSE
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
CITATION.cff
```

## Metrics sketch

- **Success rate t**: fraction of rounds breached at step t.
- **Resilience half-life**: rounds needed for success rate to drop by half after a new defense update.
- **Defense delta**: counterfactual replays of earlier prompts with current defense to estimate true learning.
- **ELO**: adversaries and defenses receive ratings based on outcomes across many loops.

## Cite

If this repository helps your research, please cite using `CITATION.cff`.
