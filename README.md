# Adaptive Adversarial Looping (AAL)

AAL is a research-grade evaluation harness that pits an adaptive adversary against a learning defender across repeated rounds. Instead of one-shot red teaming, AAL measures how models and defenses *learn under attack* — tracking breach rates, resilience decay, and ELO over time across any model you can call via API.

**Why it's different**
- Adversary adapts using a multi-armed bandit (Thompson sampling, 9 attack operators) — it learns what works against each model.
- Defender updates its prompt strategy based on previous failures.
- Metrics focus on dynamics, not snapshots: resilience half-life, defense delta, regret-conditioned ELO.
- Produces a reproducible JSONL trajectory for every run — full attack/defense/output history.

> **Safe by design:** default tasks are harmless secret-classification simulators. No real harmful content is generated or requested.

---

## Quick start

Requires Python 3.10–3.13. No Git Bash required — runs natively on Windows cmd, PowerShell, Linux, and macOS.

**Windows (cmd):**
```cmd
python -m venv .venv && .venv\Scripts\activate.bat
pip install -U pip && pip install -e .
aal run --rounds 20 --model dummy --out ./runs/smoke
aal report --in ./runs/smoke
```

**Windows (PowerShell):**
```powershell
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -U pip && pip install -e .
aal run --rounds 20 --model dummy --out ./runs/smoke
aal report --in ./runs/smoke
```

**Linux / macOS:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -e .
aal run --rounds 20 --model dummy --out ./runs/smoke
aal report --in ./runs/smoke
```

---

## Running against real models

Set your API key as an environment variable, then run. Each provider uses its own key:

**Windows (cmd):**
```cmd
set ANTHROPIC_API_KEY=your_key_here
aal run --rounds 50 --model claude --out ./runs/claude-haiku

set GROQ_API_KEY=your_key_here
aal run --rounds 50 --model groq --out ./runs/groq

set MISTRAL_API_KEY=your_key_here
aal run --rounds 50 --model mistral --out ./runs/mistral

set GEMINI_API_KEY=your_key_here
aal run --rounds 50 --model gemini --out ./runs/gemini

set OPENAI_API_KEY=your_key_here
aal run --rounds 50 --model openai --out ./runs/openai

set XAI_API_KEY=your_key_here
aal run --rounds 50 --model grok --out ./runs/grok
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY='your_key_here'
$env:GROQ_API_KEY='your_key_here'
# etc.
```

**Linux / macOS:**
```bash
export ANTHROPIC_API_KEY=your_key_here
export GROQ_API_KEY=your_key_here
# etc.
```

To run a specific Claude model by ID:
```cmd
aal run --rounds 50 --model claude-sonnet-4-6 --out ./runs/claude-sonnet
aal run --rounds 50 --model claude-opus-4-6 --out ./runs/claude-opus
```

### Supported models

| Alias | Provider | Model | Key env var |
|-------|----------|-------|-------------|
| `dummy` | Local | No API call | — |
| `claude` | Anthropic | Claude Haiku 4.5 | `ANTHROPIC_API_KEY` |
| `claude-<model-id>` | Anthropic | Any Claude model | `ANTHROPIC_API_KEY` |
| `groq` | Groq | Llama 3.1 8B | `GROQ_API_KEY` |
| `groq-llama-70b` | Groq | Llama 3.3 70B | `GROQ_API_KEY` |
| `groq-qwen` | Groq | Qwen3 32B | `GROQ_API_KEY` |
| `groq-llama4` | Groq | Llama 4 Scout 17B | `GROQ_API_KEY` |
| `groq-gpt-20b` | Groq | GPT-OSS 20B | `GROQ_API_KEY` |
| `groq-gpt-120b` | Groq | GPT-OSS 120B | `GROQ_API_KEY` |
| `mistral` | Mistral | Mistral Small | `MISTRAL_API_KEY` |
| `mistral-medium` | Mistral | Mistral Medium | `MISTRAL_API_KEY` |
| `mistral-large` | Mistral | Mistral Large | `MISTRAL_API_KEY` |
| `gemini` | Google | Gemini 2.5 Flash | `GEMINI_API_KEY` |
| `openai` | OpenAI | GPT-4.1 Nano | `OPENAI_API_KEY` |
| `openai-mini` | OpenAI | GPT-4o Mini | `OPENAI_API_KEY` |
| `openai-4o` | OpenAI | GPT-4o | `OPENAI_API_KEY` |
| `grok` | xAI | Grok 3 Mini | `XAI_API_KEY` |
| `grok-3` | xAI | Grok 3 | `XAI_API_KEY` |
| `deepseek` | DeepSeek | DeepSeek Chat | `DEEPSEEK_API_KEY` |
| `openrouter` | OpenRouter | Llama 3.3 70B (free) | `OPENROUTER_API_KEY` |
| `or-deepseek` | OpenRouter | DeepSeek R1 (free) | `OPENROUTER_API_KEY` |
| `or-gemma` | OpenRouter | Gemma 3 27B (free) | `OPENROUTER_API_KEY` |

---

## Commands

```cmd
# Run a loop
aal run --rounds 50 --model groq --out ./runs/groq

# Use a config file (CLI args override)
aal run --config examples/configs/quick.yaml --rounds 50

# View results
aal report --in ./runs/groq
aal dashboard --in ./runs/groq
aal leaderboard ./runs
aal leaderboard ./runs --last 50    # trim if a run was repeated to same folder
aal compare ./runs/groq ./runs/mistral

# Generate visual report (requires matplotlib, numpy)
python make_report.py               # outputs runs/aal_report.png
```

---

## Concepts

**Loop** — repeated rounds where the adversary proposes an attack, the defender wraps the prompt, the model responds, and a scorer assesses whether the defense held.

**Adaptive adversary** — Thompson-sampling bandit with 9 operators: translation twist, role confusion, encoding noise, flattery anchor, persona hijack, nested roleplay, authority escalation, context overflow, hypothetical frame. Allocates more attempts to operators that are working.

**Learning defender** — self-reflection strategy that updates the defense prompt based on breach history.

**Metrics**
- **Breach rate** — fraction of rounds where the adversary succeeded.
- **Resilience half-life** — rounds needed for breach rate to drop by half after a defense update.
- **Defense delta** — estimated true learning from replaying earlier prompts with the current defense.
- **ELO** — both sides receive ratings updated each round; high defender ELO means the model held under sustained pressure.

---

## Repository layout

```
aal/
  adversaries/          — attack policies (bandit baseline)
  core/                 — loop engine and types
  defenders/            — defense policies (self-reflection baseline)
  metrics/              — breach rate, half-life, delta, ELO
  models/               — provider adapters (Claude, Groq, Mistral, Gemini, OpenAI, Grok, DeepSeek, OpenRouter)
  safety/               — scorer and content policy
  storage/              — JSONL trajectory writer
  cli.py                — command-line interface
examples/configs/       — example YAML configs
make_report.py          — multi-panel visual report generator
tests/
docs/
```

---

## Running tests

```cmd
pytest -q
```

---

## Cite

If this work helps your research, please cite using `CITATION.cff`.
