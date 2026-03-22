"""Rich terminal dashboard for AAL run results.

Displays a comprehensive view of a completed run including:
- Summary statistics
- Per-operator ELO and success breakdown
- ASCII learning curve
- Defense delta and resilience half-life
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .core.types import Step
from .metrics.basic import success_rate, resilience_half_life, defense_delta, learning_curve
from .metrics.elo import compute_elo, per_op_stats
from .storage.trajectory import load_trajectory


def _ascii_chart(values: List[float], width: int = 50, height: int = 8) -> str:
    """Render a list of floats as a compact ASCII line chart."""
    if not values:
        return "(no data)"

    # Downsample to fit width
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    min_v, max_v = 0.0, 1.0
    rows: List[List[str]] = [[" "] * len(values) for _ in range(height)]

    for col, v in enumerate(values):
        row = int((1.0 - v) * (height - 1))
        row = max(0, min(height - 1, row))
        rows[row][col] = "*"

    lines = ["".join(r) for r in rows]
    lines.append("-" * len(values))
    lines.append(f"0{' ' * (len(values) - 2)}n")
    return "\n".join(lines)


def render(history: List[Step], run_name: str = "run") -> str:
    """Build a full text dashboard for *history*.

    Parameters
    ----------
    history:
        Ordered list of :class:`~aal.core.types.Step` objects.
    run_name:
        Label shown in the header.

    Returns
    -------
    str
        Multi-line dashboard string ready for printing.
    """
    if not history:
        return "No data to display."

    n = len(history)
    sr = success_rate(history)
    hl = resilience_half_life(history)
    dd = defense_delta(history)
    curve = learning_curve(history, window=max(1, n // 10))
    op_stats = per_op_stats(history)
    elos = compute_elo(history)

    lines: List[str] = []

    # Header
    lines += [
        "+" + "=" * 62 + "+",
        f"|  AAL Dashboard -- {run_name:<44}|",
        "+" + "=" * 62 + "+",
        "",
    ]

    # Summary
    lines += [
        "  SUMMARY",
        "  " + "-" * 37,
        f"  Rounds           : {n}",
        f"  Overall breach % : {sr * 100:.1f}%",
        f"  Defense ELO      : {elos.get('defender', 1200):.0f}",
        f"  Defense delta    : {dd:+.3f}  {'(v) defender improved' if dd < 0 else '(^) defender degraded' if dd > 0 else '(=) no change'}",
        f"  Resilience hl    : {'round ' + str(hl) if hl else 'not reached'}",
        "",
    ]

    # Per-operator table
    lines += [
        "  OPERATOR BREAKDOWN",
        "  " + "-" * 59,
        f"  {'Operator':<25} {'Attempts':>8} {'Breaches':>8} {'Rate':>6} {'ELO':>7}",
        "  " + "-" * 59,
    ]
    sorted_ops = sorted(op_stats.items(), key=lambda x: x[1]["elo"], reverse=True)
    for op, s in sorted_ops:
        bar = "#" * int(s["rate"] * 10) + "." * (10 - int(s["rate"] * 10))
        lines.append(
            f"  {op:<25} {s['attempts']:>8} {s['successes']:>8} "
            f" {bar} {s['rate'] * 100:>4.0f}%  {s['elo']:>7.0f}"
        )
    lines.append("")

    # Learning curve
    lines += [
        "  ADVERSARY BREACH RATE OVER TIME  (1.0 = always breaching)",
        "  +" + "-" * 52 + "+",
    ]
    chart = _ascii_chart(curve, width=50, height=6)
    for row in chart.splitlines():
        lines.append("  | " + row.ljust(50) + " |")
    lines += ["  +" + "-" * 52 + "+", ""]

    # Top breach examples
    breaches = [s for s in history if s.success]
    if breaches:
        lines += [
            "  SAMPLE BREACHES (first 3)",
            "  " + "-" * 37,
        ]
        for b in breaches[:3]:
            out_snippet = b.model_output[:80].replace("\n", " ")
            lines.append(f"  Round {b.round_id:>3}  [{b.adversary_op}]  -> {out_snippet}")
        lines.append("")

    return "\n".join(lines)


def print_dashboard(run_dir: str) -> None:
    """Load a trajectory from *run_dir* and print the dashboard."""
    path = Path(run_dir) / "trajectory.jsonl"
    if not path.exists():
        print(f"No trajectory found at {path}")
        return
    history = load_trajectory(str(path))
    run_name = Path(run_dir).name
    print(render(history, run_name=run_name))
