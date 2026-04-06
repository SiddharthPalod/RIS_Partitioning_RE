"""Shared figure annotations (e.g. model-recommended RIS partition cards)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def get_partition_stats(prefix: str, base: Path) -> dict[str, Any]:
    """Tail-window mean fractions from summary JSON, else from last segment of partitions.npy."""
    summary_path = base / f"{prefix}_summary.json"
    an = af = at = None
    L: int | None = None
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        an = s.get("a_n_mean")
        af = s.get("a_f_mean")
        at = s.get("a_t_mean")
        Lv = s.get("L")
        if Lv is not None:
            L = int(Lv)
    if an is None or af is None or at is None:
        p = base / f"{prefix}_partitions.npy"
        if p.is_file():
            arr = np.load(p)
            if len(arr) > 0:
                tail = arr[-min(500, len(arr)) :]
                an = float(tail[:, 0].mean())
                af = float(tail[:, 1].mean())
                at = float(tail[:, 2].mean())
    return {"a_n": an, "a_f": af, "a_t": at, "L": L}


def _hamilton_counts(an: float, af: float, at: float, L: int) -> tuple[int, int, int]:
    """Largest-remainder allocation of L elements to three fractions (same spirit as the env)."""
    if L <= 0:
        return 0, 0, 0
    raw = np.array([an, af, at], dtype=np.float64) * L
    floors = np.floor(raw).astype(np.int64)
    rem = int(L - int(np.sum(floors)))
    fracs = raw - floors.astype(np.float64)
    order = np.argsort(-fracs)
    extras = np.zeros(3, dtype=np.int64)
    for k in range(rem):
        extras[order[k]] += 1
    c = floors + extras
    return int(c[0]), int(c[1]), int(c[2])


def format_optimal_partition_card(
    base: Path,
    prefix_dqn: str,
    prefix_ddpg: str,
    *,
    title: str = "Model-recommended RIS partition (tail window)",
) -> str:
    """
    Text block: optimal fractions a_n, a_f, a_t and approximate element counts L_n, L_f, L_t when L is known.
    """
    lines = [title, ""]
    for algo, pref in [("DQN", prefix_dqn), ("DDPG", prefix_ddpg)]:
        st = get_partition_stats(pref, base)
        an, af, at = st["a_n"], st["a_f"], st["a_t"]
        L = st.get("L")
        if an is None or af is None or at is None:
            lines.append(f"{algo}: (no partition log)")
            lines.append("")
            continue
        lines.append(f"{algo}:")
        lines.append(f"  shares  a_n = {an:.4f}   a_f = {af:.4f}   a_t = {at:.4f}")
        if L is not None and L > 0:
            ln, lf, lt = _hamilton_counts(an, af, at, L)
            lines.append(f"  elements  L_n = {ln} ,  L_f = {lf} ,  L_t = {lt}   (L = {L})")
        else:
            lines.append("  elements  (set env L in summary to show L_n, L_f, L_t)")
        lines.append("")
    return "\n".join(lines).rstrip()


def add_partition_card(
    fig,
    base: Path,
    prefix_dqn: str,
    prefix_ddpg: str,
    *,
    title: str | None = None,
    loc: str = "upper right",
    fontsize: float = 7.5,
) -> None:
    """Draw a rounded text card on a matplotlib Figure (uses transFigure)."""
    text = format_optimal_partition_card(
        base,
        prefix_dqn,
        prefix_ddpg,
        title=title or "Model-recommended RIS partition (tail window)",
    )
    if loc == "upper right":
        x, y, ha, va = 0.98, 0.98, "right", "top"
    elif loc == "lower right":
        x, y, ha, va = 0.98, 0.02, "right", "bottom"
    elif loc == "upper left":
        x, y, ha, va = 0.02, 0.98, "left", "top"
    else:
        x, y, ha, va = 0.02, 0.02, "left", "bottom"

    fig.text(
        x,
        y,
        text,
        transform=fig.transFigure,
        ha=ha,
        va=va,
        fontsize=fontsize,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "0.65",
            "alpha": 0.96,
            "linewidth": 0.8,
        },
        zorder=10,
    )
