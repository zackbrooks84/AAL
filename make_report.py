"""
AAL Visual Report — generates a multi-panel PNG from completed run data.
Run from the AAL root: python make_report.py
Output: runs/aal_report.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

# Known models with distinct colors — each hue is unique
KNOWN_RUNS = {
    "mistral":          {"label": "Mistral Small",        "color": "#2ecc71"},  # green
    "mistral-medium":   {"label": "Mistral Medium",       "color": "#1abc9c"},  # teal
    "mistral-large":    {"label": "Mistral Large",        "color": "#f1c40f"},  # yellow
    "groq":             {"label": "Groq Llama 8B",        "color": "#3498db"},  # blue
    "groq-70b":         {"label": "Groq Llama 70B",       "color": "#9b59b6"},  # purple
    "groq-qwen":        {"label": "Groq Qwen3 32B",       "color": "#e67e22"},  # orange
    "groq-llama4":      {"label": "Groq Llama 4 Scout",   "color": "#00bcd4"},  # cyan
    "groq-gpt-20b":     {"label": "Groq GPT-OSS 20B",     "color": "#e91e63"},  # pink
    "groq-gpt-120b":    {"label": "Groq GPT-OSS 120B",    "color": "#ff5722"},  # deep orange
    "gemini":           {"label": "Gemini 2.5 Flash",     "color": "#e74c3c"},  # red
    "deepseek":         {"label": "DeepSeek Chat",        "color": "#8bc34a"},  # lime
    "openrouter":       {"label": "OpenRouter Llama 70B", "color": "#ff9800"},  # amber
    "openai":           {"label": "OpenAI GPT-4.1 Nano",  "color": "#b0bec5"},  # silver
    "openai-mini":      {"label": "OpenAI GPT-4o Mini",   "color": "#ecf0f1"},  # white
    "openai-4o":        {"label": "OpenAI GPT-4o",        "color": "#ffffff"},  # bright white
    "grok":             {"label": "Grok 3 Mini",          "color": "#a78bfa"},  # violet
    "grok-3":           {"label": "Grok 3",               "color": "#7c3aed"},  # deep violet
    "grok-2":           {"label": "Grok 2",               "color": "#c4b5fd"},  # light violet
}

# Fallback palette for any run folder not in KNOWN_RUNS
_FALLBACK_COLORS = [
    "#b0bec5", "#a1887f", "#4db6ac", "#dce775",
    "#ba68c8", "#4fc3f7", "#ffb74d", "#aed581",
]

# Auto-discover all run folders that have a trajectory
_runs_dir = Path("runs")
_auto_color_idx = 0
RUNS = {}
for _p in sorted(_runs_dir.iterdir()):
    if not (_p / "trajectory.jsonl").exists():
        continue
    if _p.name in KNOWN_RUNS:
        RUNS[_p.name] = KNOWN_RUNS[_p.name]
    else:
        RUNS[_p.name] = {
            "label": _p.name,
            "color": _FALLBACK_COLORS[_auto_color_idx % len(_FALLBACK_COLORS)],
        }
        _auto_color_idx += 1

OPS = [
    "translation_twist", "role_confusion", "encoding_noise",
    "flattery_anchor", "persona_hijack", "nested_roleplay",
    "authority_escalation", "context_overflow", "hypothetical_frame",
]

OP_SHORT = {
    "translation_twist":   "translation",
    "role_confusion":      "role_confuse",
    "encoding_noise":      "encoding",
    "flattery_anchor":     "flattery",
    "persona_hijack":      "persona",
    "nested_roleplay":     "roleplay",
    "authority_escalation":"authority",
    "context_overflow":    "ctx_overflow",
    "hypothetical_frame":  "hypothetical",
}

data = {}
for name in RUNS:
    p = Path(f"runs/{name}/trajectory.jsonl")
    steps = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if steps:
        data[name] = steps

# ---------------------------------------------------------------------------
# Derived stats
# ---------------------------------------------------------------------------

def breach_rate(steps):
    return sum(1 for s in steps if s["success"]) / len(steps)

def rolling_breach(steps, window=5):
    rates = []
    for i in range(len(steps)):
        w = steps[max(0, i - window + 1): i + 1]
        rates.append(sum(1 for s in w if s["success"]) / len(w))
    return rates

def op_rates(steps):
    counts = {op: {"n": 0, "breaches": 0} for op in OPS}
    for s in steps:
        op = s["adversary_op"]
        if op in counts:
            counts[op]["n"] += 1
            if s["success"]:
                counts[op]["breaches"] += 1
    return {op: (v["breaches"] / v["n"] if v["n"] > 0 else 0) for op, v in counts.items()}

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(18, 14), facecolor="#0f1117")
fig.suptitle(
    "AAL Adversarial Robustness Report",
    fontsize=20, fontweight="bold", color="white", y=0.98
)

gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
              left=0.07, right=0.97, top=0.93, bottom=0.06)

ax_bar    = fig.add_subplot(gs[0, :2])   # breach rate bar chart
ax_elo    = fig.add_subplot(gs[0, 2])    # defender ELO
ax_curve  = fig.add_subplot(gs[1, :])    # learning curves
ax_heatmap = fig.add_subplot(gs[2, :])  # operator heatmap

DARK  = "#0f1117"
CARD  = "#1a1d2e"
GRID  = "#2a2d3e"
TEXT  = "#e0e0e0"
SUB   = "#888888"

for ax in [ax_bar, ax_elo, ax_curve, ax_heatmap]:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

# ---------------------------------------------------------------------------
# Panel 1 — Overall breach rate bar chart
# ---------------------------------------------------------------------------

names   = list(data.keys())
labels  = [RUNS[n]["label"] for n in names]
colors  = [RUNS[n]["color"] for n in names]
rates   = [breach_rate(data[n]) * 100 for n in names]

order   = sorted(range(len(rates)), key=lambda i: rates[i])
names   = [names[i] for i in order]
labels  = [labels[i] for i in order]
colors  = [colors[i] for i in order]
rates   = [rates[i] for i in order]

bars = ax_bar.barh(labels, rates, color=colors, height=0.55, zorder=2)
ax_bar.set_xlim(0, 110)
ax_bar.set_xlabel("Breach Rate (%)", color=TEXT, fontsize=10)
ax_bar.set_title("Overall Breach Rate  (lower = more robust)", color=TEXT, fontsize=11, pad=8)
ax_bar.axvline(50, color=GRID, linewidth=1, linestyle="--", zorder=1)
ax_bar.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
ax_bar.set_axisbelow(True)

for bar, rate in zip(bars, rates):
    ax_bar.text(rate + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.0f}%", va="center", color=TEXT, fontsize=10, fontweight="bold")

# ---------------------------------------------------------------------------
# Panel 2 — Defender ELO
# ---------------------------------------------------------------------------

def elo_scores(steps, k=32, base=1200):
    defender_elo = base
    for s in steps:
        adv_elo = base
        e_d = 1 / (1 + 10 ** ((adv_elo - defender_elo) / 400))
        outcome = 0 if s["success"] else 1
        defender_elo += k * (outcome - e_d)
    return defender_elo

elo_names  = [RUNS[n]["label"] for n in names]
elo_vals   = [elo_scores(data[n]) for n in names]
elo_colors = [RUNS[n]["color"] for n in names]

ax_elo.barh(elo_names, elo_vals, color=elo_colors, height=0.55, zorder=2)
ax_elo.axvline(1200, color="#888888", linewidth=1, linestyle="--", zorder=1)
ax_elo.set_xlabel("ELO Rating", color=TEXT, fontsize=10)
ax_elo.set_title("Defender ELO\n(higher = stronger defense)", color=TEXT, fontsize=11, pad=8)
ax_elo.xaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
ax_elo.set_axisbelow(True)
for i, v in enumerate(elo_vals):
    ax_elo.text(v + 5, i, f"{v:.0f}", va="center", color=TEXT, fontsize=9)

# ---------------------------------------------------------------------------
# Panel 3 — Rolling breach rate (learning curves)
# ---------------------------------------------------------------------------

ax_curve.set_title("Breach Rate Over Time  (rolling 5-round window)", color=TEXT, fontsize=11, pad=8)
ax_curve.set_xlabel("Round", color=TEXT, fontsize=10)
ax_curve.set_ylabel("Breach Rate", color=TEXT, fontsize=10)
ax_curve.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
ax_curve.set_axisbelow(True)
ax_curve.set_ylim(-0.05, 1.15)

for name in data:
    curve = rolling_breach(data[name], window=5)
    x = list(range(1, len(curve) + 1))
    ax_curve.plot(x, curve, color=RUNS[name]["color"], linewidth=2,
                  label=RUNS[name]["label"], zorder=2)
    ax_curve.fill_between(x, curve, alpha=0.08, color=RUNS[name]["color"])

ax_curve.axhline(0.5, color="#555555", linewidth=1, linestyle=":", zorder=1)
ax_curve.legend(loc="upper right", facecolor=CARD, edgecolor=GRID,
                labelcolor=TEXT, fontsize=9)

# ---------------------------------------------------------------------------
# Panel 4 — Operator heatmap
# ---------------------------------------------------------------------------

ax_heatmap.set_title("Attack Operator Success Rate by Model  (red = adversary wins)", color=TEXT, fontsize=11, pad=8)

model_labels = [RUNS[n]["label"] for n in data]
op_labels    = [OP_SHORT[op] for op in OPS]
matrix       = np.array([[op_rates(data[n]).get(op, 0) for op in OPS] for n in data])

im = ax_heatmap.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

ax_heatmap.set_xticks(range(len(OPS)))
ax_heatmap.set_xticklabels(op_labels, rotation=30, ha="right", color=TEXT, fontsize=9)
ax_heatmap.set_yticks(range(len(model_labels)))
ax_heatmap.set_yticklabels(model_labels, color=TEXT, fontsize=9)

for i in range(len(model_labels)):
    for j in range(len(OPS)):
        val = matrix[i, j]
        txt_color = "white" if val > 0.6 or val < 0.2 else "black"
        ax_heatmap.text(j, i, f"{val:.0%}", ha="center", va="center",
                        color=txt_color, fontsize=8, fontweight="bold")

cbar = fig.colorbar(im, ax=ax_heatmap, orientation="vertical", pad=0.01, shrink=0.9)
cbar.ax.tick_params(colors=TEXT, labelsize=8)
cbar.set_label("Breach Rate", color=TEXT, fontsize=9)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

fig.text(0.5, 0.01,
         "Emergence Archive  |  AAL v0.1  |  Adaptive Adversarial Looping  |  March 30, 2026",
         ha="center", color=SUB, fontsize=8)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out = Path("runs/aal_report.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"Saved: {out.resolve()}")
