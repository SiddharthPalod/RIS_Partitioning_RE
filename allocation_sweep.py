"""
Brute-force sweep over RIS allocation simplex (a_n, a_f, a_t) with a_n+a_f+a_t=1.

This diagnoses whether the environment/reward itself prefers near-heavy, far-heavy,
or sensing-heavy partitions, independent of RL optimization dynamics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config.paths import RESULTS_DIR, ensure_results_dir
from env.isac_env import ISACRISEnv


def _build_simplex(step: float) -> list[tuple[float, float, float]]:
    pts: list[tuple[float, float, float]] = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        a_n = i * step
        for j in range(n + 1 - i):
            a_f = j * step
            a_t = 1.0 - a_n - a_f
            if a_t < -1e-12:
                continue
            # Numerical cleanup so columns/labels look stable.
            pts.append((round(a_n, 10), round(a_f, 10), round(max(0.0, a_t), 10)))
    return pts


def _eval_point(env: ISACRISEnv, alloc: tuple[float, float, float], mc: int) -> dict[str, float]:
    a_n, a_f, a_t = alloc
    rn_l: list[float] = []
    rf_l: list[float] = []
    asir_l: list[float] = []
    jfi_l: list[float] = []
    reward_l: list[float] = []

    for _ in range(mc):
        env.a_n, env.a_f, env.a_t = a_n, a_f, a_t
        env._sample_channels()
        rn, rf, asir, jfi, reward, *_ = env._compute_metrics()
        rn_l.append(rn)
        rf_l.append(rf)
        asir_l.append(asir)
        jfi_l.append(jfi)
        reward_l.append(reward)

    rn_m = float(np.mean(rn_l))
    rf_m = float(np.mean(rf_l))
    asir_m = float(np.mean(asir_l))
    jfi_m = float(np.mean(jfi_l))
    reward_m = float(np.mean(reward_l))
    return {
        "a_n": float(a_n),
        "a_f": float(a_f),
        "a_t": float(a_t),
        "r_n_mean": rn_m,
        "r_f_mean": rf_m,
        "rsum_mean": float(rn_m + rf_m),
        "asir_mean": asir_m,
        "jfi_mean": jfi_m,
        "reward_mean": reward_m,
    }


def _save_csv(rows: list[dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "a_n",
        "a_f",
        "a_t",
        "r_n_mean",
        "r_f_mean",
        "rsum_mean",
        "asir_mean",
        "jfi_mean",
        "reward_mean",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _save_plot(rows: list[dict[str, float]], out_png: Path) -> None:
    a_n = np.array([r["a_n"] for r in rows], dtype=np.float64)
    a_f = np.array([r["a_f"] for r in rows], dtype=np.float64)
    reward = np.array([r["reward_mean"] for r in rows], dtype=np.float64)
    jfi = np.array([r["jfi_mean"] for r in rows], dtype=np.float64)
    rsum = np.array([r["rsum_mean"] for r in rows], dtype=np.float64)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    for i, (vals, title) in enumerate(
        [
            (reward, "Mean Reward"),
            (rsum, "Mean Sum-Rate (R_n + R_f)"),
            (jfi, "Mean JFI"),
        ]
    ):
        sc = ax[i].scatter(a_n, a_f, c=vals, cmap="viridis", s=34, edgecolors="none")
        ax[i].set_title(title)
        ax[i].set_xlabel("a_n")
        ax[i].set_ylabel("a_f")
        ax[i].grid(alpha=0.25)
        plt.colorbar(sc, ax=ax[i], fraction=0.046, pad=0.04)

    fig.suptitle("Allocation Sweep on Simplex (a_t = 1 - a_n - a_f)")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brute-force allocation sweep on RIS simplex.")
    parser.add_argument("--step", type=float, default=0.05, help="Simplex grid step (e.g., 0.05).")
    parser.add_argument("--mc", type=int, default=300, help="Monte Carlo channel draws per grid point.")
    parser.add_argument("--w-c", type=float, default=0.55, dest="w_c")
    parser.add_argument("--w-r", type=float, default=0.25, dest="w_r")
    parser.add_argument("--w-f", type=float, default=0.20, dest="w_f")
    parser.add_argument("--w-qos-f", type=float, default=0.0, dest="w_qos_f")
    parser.add_argument("--w-qos-s", type=float, default=0.0, dest="w_qos_s")
    parser.add_argument("--rf-min", type=float, default=0.0, dest="rf_min")
    parser.add_argument("--asir-min", type=float, default=0.0, dest="asir_min")
    parser.add_argument("--max-rsum", type=float, default=1.2e-3, dest="max_rsum")
    parser.add_argument("--max-asir", type=float, default=8.5e-4, dest="max_asir")
    parser.add_argument("--adaptive-norm", action="store_true", default=False, help="Enable adaptive normalization during sweep.")
    parser.add_argument("--L", type=int, default=1000)
    parser.add_argument("--pn-dbm", type=float, default=20.0, dest="pn_dbm")
    parser.add_argument("--pt-dbm", type=float, default=30.0, dest="pt_dbm")
    parser.add_argument("--prefix", type=str, default="allocation_sweep")
    args = parser.parse_args()

    ensure_results_dir()
    env = ISACRISEnv(
        L=args.L,
        Pn_dBm=args.pn_dbm,
        Pt_dBm=args.pt_dbm,
        w_c=args.w_c,
        w_r=args.w_r,
        w_f=args.w_f,
        w_qos_f=args.w_qos_f,
        w_qos_s=args.w_qos_s,
        rf_min=args.rf_min,
        asir_min=args.asir_min,
        max_rsum=args.max_rsum,
        max_asir=args.max_asir,
        adaptive_norm=args.adaptive_norm,
    )
    env.reset()

    pts = _build_simplex(args.step)
    rows = [_eval_point(env, alloc, args.mc) for alloc in pts]
    rows.sort(key=lambda r: r["reward_mean"], reverse=True)

    out_csv = RESULTS_DIR / f"{args.prefix}.csv"
    out_png = RESULTS_DIR / f"{args.prefix}.png"
    _save_csv(rows, out_csv)
    _save_plot(rows, out_png)

    best = rows[0]
    print(
        "Best reward allocation: "
        f"a=[{best['a_n']:.3f},{best['a_f']:.3f},{best['a_t']:.3f}] | "
        f"reward={best['reward_mean']:.6f} | rsum={best['rsum_mean']:.6e} | "
        f"jfi={best['jfi_mean']:.4f} | asir={best['asir_mean']:.6e}"
    )
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote Figure: {out_png}")


if __name__ == "__main__":
    main()

