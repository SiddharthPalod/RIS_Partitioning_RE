"""
Train DQN and DDPG with the same seed and reward weights, then plot reward / JFI / (R_n + R_f).
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
from figure_utils import add_partition_card
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
    p_dqn = np.load(base / f"{prefix_dqn}_partitions.npy")
    rn_dqn = np.load(base / f"{prefix_dqn}_r1.npy")
    rf_dqn = np.load(base / f"{prefix_dqn}_r2.npy")
    asir_dqn = np.load(base / f"{prefix_dqn}_asir.npy")

    r_ddpg = np.load(base / f"{prefix_ddpg}_rewards.npy")
    j_ddpg = np.load(base / f"{prefix_ddpg}_jfi.npy")
    p_ddpg = np.load(base / f"{prefix_ddpg}_partitions.npy")
    rn_ddpg = np.load(base / f"{prefix_ddpg}_r1.npy")
    rf_ddpg = np.load(base / f"{prefix_ddpg}_r2.npy")
    asir_ddpg = np.load(base / f"{prefix_ddpg}_asir.npy")

    n = min(len(r_dqn), len(r_ddpg))
    r_dqn, j_dqn = r_dqn[:n], j_dqn[:n]
    r_ddpg, j_ddpg = r_ddpg[:n], j_ddpg[:n]
    p_dqn, rn_dqn, rf_dqn, asir_dqn = (
        p_dqn[:n],
        rn_dqn[:n],
        rf_dqn[:n],
        asir_dqn[:n],
    )
    p_ddpg, rn_ddpg, rf_ddpg, asir_ddpg = (
        p_ddpg[:n],
        rn_ddpg[:n],
        rf_ddpg[:n],
        asir_ddpg[:n],
    )

    rsum_dqn = rn_dqn + rf_dqn
    rsum_ddpg = rn_ddpg + rf_ddpg

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
        (axes[0], r_dqn, r_ddpg, "Reward", r"$r_t = w_c \,\widehat{(R_n+R_f)} + w_r \,\widehat{\mathrm{ASIR}} + w_f\,\mathrm{JFI}$"),
        (axes[1], j_dqn, j_ddpg, "JFI$(R_n, R_f)$", "JFI"),
        (
            axes[2],
            rsum_dqn,
            rsum_ddpg,
            r"Communication sum-rate $R_n + R_f$",
            "Sum-rate",
        ),
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
        ax.legend(loc="lower left", fontsize=8)

    plt.tight_layout(rect=[0, 0, 0.78, 1])
    add_partition_card(
        fig,
        base,
        prefix_dqn,
        prefix_ddpg,
        loc="upper right",
        fontsize=7.0,
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
        plt.close()
    else:
        plt.show()


def plot_diagnostics(
    prefix_dqn: str,
    prefix_ddpg: str,
    smooth_window: int,
    out_path: Optional[Path],
    *,
    rf_min: float = 0.0,
    asir_min: float = 0.0,
) -> None:
    """QoS-relevant diagnostics: rates, sensing, partitions, and violation trends."""
    base = RESULTS_DIR
    rn_dqn = np.load(base / f"{prefix_dqn}_r1.npy")
    rf_dqn = np.load(base / f"{prefix_dqn}_r2.npy")
    asir_dqn = np.load(base / f"{prefix_dqn}_asir.npy")
    p_dqn = np.load(base / f"{prefix_dqn}_partitions.npy")

    rn_ddpg = np.load(base / f"{prefix_ddpg}_r1.npy")
    rf_ddpg = np.load(base / f"{prefix_ddpg}_r2.npy")
    asir_ddpg = np.load(base / f"{prefix_ddpg}_asir.npy")
    p_ddpg = np.load(base / f"{prefix_ddpg}_partitions.npy")

    n = min(len(rn_dqn), len(rn_ddpg))
    rn_dqn, rf_dqn, asir_dqn, p_dqn = rn_dqn[:n], rf_dqn[:n], asir_dqn[:n], p_dqn[:n]
    rn_ddpg, rf_ddpg, asir_ddpg, p_ddpg = rn_ddpg[:n], rf_ddpg[:n], asir_ddpg[:n], p_ddpg[:n]
    x = np.arange(n)

    def smooth(y: np.ndarray) -> np.ndarray:
        return ema_filter(y, max(2, smooth_window))

    # Instantaneous QoS violations (1 if violated, else 0), then smooth.
    rf_v_dqn = (rf_dqn < float(rf_min)).astype(np.float32)
    rf_v_ddpg = (rf_ddpg < float(rf_min)).astype(np.float32)
    as_v_dqn = (asir_dqn < float(asir_min)).astype(np.float32)
    as_v_ddpg = (asir_ddpg < float(asir_min)).astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1) Communication rates
    ax = axes[0, 0]
    ax.plot(x, smooth(rn_dqn), color="tab:blue", lw=1.5, label="DQN R_n")
    ax.plot(x, smooth(rf_dqn), color="tab:blue", ls="--", lw=1.3, label="DQN R_f")
    ax.plot(x, smooth(rn_ddpg), color="tab:orange", lw=1.5, label="DDPG R_n")
    ax.plot(x, smooth(rf_ddpg), color="tab:orange", ls="--", lw=1.3, label="DDPG R_f")
    if rf_min > 0:
        ax.axhline(float(rf_min), color="red", lw=1.0, ls=":", label="R_f target")
    ax.set_title("Communication Rates")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    # 2) ASIR
    ax = axes[0, 1]
    ax.plot(x, smooth(asir_dqn), color="tab:blue", lw=1.5, label="DQN ASIR")
    ax.plot(x, smooth(asir_ddpg), color="tab:orange", lw=1.5, label="DDPG ASIR")
    if asir_min > 0:
        ax.axhline(float(asir_min), color="red", lw=1.0, ls=":", label="ASIR target")
    ax.set_title("Sensing Rate (ASIR)")
    ax.set_xlabel("Step")
    ax.set_ylabel("ASIR")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    # 3) QoS violation trends (EMA of indicator)
    ax = axes[0, 2]
    ax.plot(x, smooth(rf_v_dqn), color="tab:blue", lw=1.5, label="DQN R_f violation")
    ax.plot(x, smooth(as_v_dqn), color="tab:blue", ls="--", lw=1.3, label="DQN ASIR violation")
    ax.plot(x, smooth(rf_v_ddpg), color="tab:orange", lw=1.5, label="DDPG R_f violation")
    ax.plot(x, smooth(as_v_ddpg), color="tab:orange", ls="--", lw=1.3, label="DDPG ASIR violation")
    ax.set_title("QoS Violation Trend (0 to 1)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Violation Rate (EMA)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")

    # 4) DQN partitions
    ax = axes[1, 0]
    ax.stackplot(x, p_dqn[:, 0], p_dqn[:, 1], p_dqn[:, 2], labels=[r"$a_n$", r"$a_f$", r"$a_t$"], alpha=0.85)
    ax.set_title("DQN Partition Mix")
    ax.set_xlabel("Step")
    ax.set_ylabel("Share")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="upper right")

    # 5) DDPG partitions
    ax = axes[1, 1]
    ax.stackplot(x, p_ddpg[:, 0], p_ddpg[:, 1], p_ddpg[:, 2], labels=[r"$a_n$", r"$a_f$", r"$a_t$"], alpha=0.85)
    ax.set_title("DDPG Partition Mix")
    ax.set_xlabel("Step")
    ax.set_ylabel("Share")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="upper right")

    # 6) Cumulative QoS satisfactions
    ax = axes[1, 2]
    sat_dqn = ((rf_dqn >= float(rf_min)) & (asir_dqn >= float(asir_min))).astype(np.float32)
    sat_ddpg = ((rf_ddpg >= float(rf_min)) & (asir_ddpg >= float(asir_min))).astype(np.float32)
    ax.plot(x, np.cumsum(sat_dqn), color="tab:blue", lw=1.6, label="DQN")
    ax.plot(x, np.cumsum(sat_ddpg), color="tab:orange", lw=1.6, label="DDPG")
    ax.set_title("Cumulative Joint QoS Satisfaction")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

    fig.suptitle("QoS-Focused Diagnostics", y=1.01, fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.78, 0.99])
    add_partition_card(fig, base, prefix_dqn, prefix_ddpg, loc="upper right", fontsize=7.0)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved diagnostics figure to {out_path}")
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
    parser.add_argument("--w-c", "--lambda1", type=float, default=0.55, dest="w_c", help="w_c for normalized communication term.")
    parser.add_argument("--w-r", "--lambda2", type=float, default=0.25, dest="w_r", help="w_r for normalized sensing term.")
    parser.add_argument("--w-f", type=float, default=0.20, dest="w_f", help="w_f for fairness term JFI(R_n, R_f).")
    parser.add_argument("--w-qos-f", type=float, default=0.0, dest="w_qos_f", help="Penalty weight for far-user QoS shortfall.")
    parser.add_argument("--w-qos-s", type=float, default=0.0, dest="w_qos_s", help="Penalty weight for sensing QoS shortfall.")
    parser.add_argument("--rf-min", type=float, default=0.0, dest="rf_min", help="Far-user minimum target.")
    parser.add_argument("--asir-min", type=float, default=0.0, dest="asir_min", help="Sensing minimum target.")
    parser.add_argument("--max-rsum", type=float, default=1.2e-3, dest="max_rsum", help="Normalization scale for (R_n + R_f).")
    parser.add_argument("--max-asir", type=float, default=8.5e-4, dest="max_asir", help="Normalization scale for ASIR.")
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

    env_kwargs = {
        "w_c": args.w_c,
        "w_r": args.w_r,
        "w_f": args.w_f,
        "w_qos_f": args.w_qos_f,
        "w_qos_s": args.w_qos_s,
        "rf_min": args.rf_min,
        "asir_min": args.asir_min,
        "max_rsum": args.max_rsum,
        "max_asir": args.max_asir,
    }

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
    diag_out_path: Optional[Path]
    if out_path is None:
        diag_out_path = None
    else:
        diag_out_path = out_path.with_name(f"{out_path.stem}_diagnostics{out_path.suffix}")
    plot_diagnostics(
        args.prefix_dqn,
        args.prefix_ddpg,
        smooth_window=args.ma_window,
        out_path=diag_out_path,
        rf_min=args.rf_min,
        asir_min=args.asir_min,
    )

    meta: dict[str, Any] = {
        "seed": args.seed,
        "steps": args.steps,
        "w_c": args.w_c,
        "w_r": args.w_r,
        "w_f": args.w_f,
        "w_qos_f": args.w_qos_f,
        "w_qos_s": args.w_qos_s,
        "rf_min": args.rf_min,
        "asir_min": args.asir_min,
        "max_rsum": args.max_rsum,
        "max_asir": args.max_asir,
        "prefix_dqn": args.prefix_dqn,
        "prefix_ddpg": args.prefix_ddpg,
        "no_train": args.no_train,
        "plot_path": str(out_path) if out_path else None,
        "plot_diagnostics_path": str(diag_out_path) if diag_out_path else None,
        "summary_dqn": str(RESULTS_DIR / f"{args.prefix_dqn}_summary.json"),
        "summary_ddpg": str(RESULTS_DIR / f"{args.prefix_ddpg}_summary.json"),
    }
    _write_compare_summary(meta)


if __name__ == "__main__":
    main()
