"""
Train DQN and DDPG with the same seed and reward weights, then plot reward / JFI / sum-rate.
Artifacts live under ``Paper5/results/``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from config.paths import (
    DEFAULT_LOG_PREFIX_DDPG,
    DEFAULT_LOG_PREFIX_DQN,
    RESULTS_DIR,
    ensure_results_dir,
)
from training.loops import run_ddpg, run_dqn


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x.astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x.astype(np.float64), kernel, mode="valid").astype(np.float32)


def ema_filter(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average; same length as x."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x.astype(np.float32)
    if span < 2:
        return x.astype(np.float32)
    alpha = 2.0 / (float(span) + 1.0)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y.astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_comparison(
    prefix_dqn: str,
    prefix_ddpg: str,
    smooth_window: int,
    out_path: Optional[Path],
    *,
    smooth: str = "ema",
    show_raw: bool = True,
) -> None:
    base = RESULTS_DIR
    r_dqn = np.load(base / f"{prefix_dqn}_rewards.npy")
    j_dqn = np.load(base / f"{prefix_dqn}_jfi.npy")
    s_dqn = np.load(base / f"{prefix_dqn}_rsum.npy")

    r_ddpg = np.load(base / f"{prefix_ddpg}_rewards.npy")
    j_ddpg = np.load(base / f"{prefix_ddpg}_jfi.npy")
    s_ddpg = np.load(base / f"{prefix_ddpg}_rsum.npy")

    n = min(len(r_dqn), len(r_ddpg))
    r_dqn, j_dqn, s_dqn = r_dqn[:n], j_dqn[:n], s_dqn[:n]
    r_ddpg, j_ddpg, s_ddpg = r_ddpg[:n], j_ddpg[:n], s_ddpg[:n]

    def smooth_line(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if smooth == "ema":
            ys = ema_filter(y, smooth_window)
            return np.arange(n), ys
        ma = moving_average(y, smooth_window)
        if len(ma) == len(y):
            xd = np.arange(n, dtype=np.float32)
        else:
            xd = np.arange(smooth_window - 1, n, dtype=np.float32)
        return xd, ma

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(n)

    for ax, y_dqn, y_ddpg, title, ylabel in [
        (axes[0], r_dqn, r_ddpg, "Reward", r"$r_t = \lambda_1 \mathrm{JFI} + \lambda_2 \mathrm{ASIR}$"),
        (axes[1], j_dqn, j_ddpg, "JFI$(R_n, R_f)$", "JFI"),
        (axes[2], s_dqn, s_ddpg, "Sum rate $R_n + R_f$", "Rate"),
    ]:
        if show_raw:
            ax.plot(x, y_dqn, alpha=0.25, color="tab:blue", linewidth=0.6, label="DQN (raw)")
            ax.plot(x, y_ddpg, alpha=0.25, color="tab:orange", linewidth=0.6, label="DDPG (raw)")
        sw = max(1, smooth_window)
        xd_dqn, s_dqn_y = smooth_line(y_dqn)
        xd_dp, s_ddpg_y = smooth_line(y_ddpg)
        tag = f"EMA({sw})" if smooth == "ema" else f"MA({sw})"
        ax.plot(xd_dqn, s_dqn_y, color="tab:blue", lw=1.8, label=f"DQN {tag}")
        ax.plot(xd_dp, s_ddpg_y, color="tab:orange", lw=1.8, label=f"DDPG {tag}")
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
        plt.close()
    else:
        plt.show()


def _write_compare_summary(meta: dict[str, Any]) -> None:
    ensure_results_dir()
    out = RESULTS_DIR / "compare_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out}")


def main() -> None:
    default_plot = RESULTS_DIR / "compare.png"
    parser = argparse.ArgumentParser(description="Compare DQN vs DDPG (train + plot)")
    parser.add_argument("--steps", type=int, default=8000, help="Training steps per algorithm.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (both runs).")
    parser.add_argument("--lambda1", type=float, default=0.6, help="lambda_1 for JFI.")
    parser.add_argument("--lambda2", type=float, default=0.008, help="lambda_2 for ASIR.")
    parser.add_argument(
        "--ma-window",
        type=int,
        default=200,
        help="Smoothing: EMA span (default) or causal MA window if --smooth ma.",
    )
    parser.add_argument(
        "--smooth",
        type=str,
        choices=["ema", "ma"],
        default="ema",
        help="ema=exponential moving average; ma=causal moving average.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Hide per-step raw traces; only plot smoothed curves.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Only plot from existing npy files (no training).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=f"Figure path. Default: {default_plot}. Use - or show for interactive window only.",
    )
    parser.add_argument(
        "--prefix-dqn",
        type=str,
        default=DEFAULT_LOG_PREFIX_DQN,
        help="Log filename prefix for DQN.",
    )
    parser.add_argument(
        "--prefix-ddpg",
        type=str,
        default=DEFAULT_LOG_PREFIX_DDPG,
        help="Log filename prefix for DDPG.",
    )
    args = parser.parse_args()

    env_kwargs = {"lambda_1": args.lambda1, "lambda_2": args.lambda2}

    if not args.no_train:
        set_seed(args.seed)
        run_dqn(total_steps=args.steps, log_prefix=args.prefix_dqn, env_kwargs=env_kwargs)
        set_seed(args.seed)
        run_ddpg(total_steps=args.steps, log_prefix=args.prefix_ddpg, env_kwargs=env_kwargs)

    out_arg = (args.out or "").strip().lower()
    if out_arg in ("-", "show", "none"):
        out_path: Optional[Path] = None
    elif args.out.strip():
        p = Path(args.out.strip())
        out_path = p if p.is_absolute() else RESULTS_DIR / p.name
    else:
        out_path = default_plot

    plot_comparison(
        args.prefix_dqn,
        args.prefix_ddpg,
        smooth_window=args.ma_window,
        out_path=out_path,
        smooth=args.smooth,
        show_raw=not args.no_raw,
    )

    meta: dict[str, Any] = {
        "seed": args.seed,
        "steps": args.steps,
        "lambda_1": args.lambda1,
        "lambda_2": args.lambda2,
        "prefix_dqn": args.prefix_dqn,
        "prefix_ddpg": args.prefix_ddpg,
        "no_train": args.no_train,
        "plot_path": str(out_path) if out_path else None,
        "summary_dqn": str(RESULTS_DIR / f"{args.prefix_dqn}_summary.json"),
        "summary_ddpg": str(RESULTS_DIR / f"{args.prefix_ddpg}_summary.json"),
    }
    _write_compare_summary(meta)


if __name__ == "__main__":
    main()
