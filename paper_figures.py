"""
Generate publication-style figures from ``results/*.npy`` (after training).

Figures (saved under ``results/figures/`` by default):
  - figure5_convergence_episodes.png — mean reward per episode (DRL convergence)
  - figure6_cumulative_rewards.png — cumulative reward over steps for multiple runs
  - figure7_rates_dqn_ddpg.png — near/far/sum rates + ergodic (rolling mean) DQN vs DDPG
  - figure8_hybrid_vs_passive.png — hybrid ISAC sum rate vs passive-NOMA-only baseline
  - figure9_losses.png — training losses (diagnostic)

Requires ``training_*_summary.json`` for ``max_steps`` / ``L`` when present; else defaults 30 / 1000.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config.paths import DEFAULT_LOG_PREFIX_DDPG, DEFAULT_LOG_PREFIX_DQN, RESULTS_DIR, ensure_results_dir
from figure_utils import add_partition_card


def ema_filter(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0 or span < 2:
        return x.astype(np.float32)
    alpha = 2.0 / (float(span) + 1.0)
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y.astype(np.float32)


def _load_summary(prefix: str, base: Path) -> dict[str, Any]:
    p = base / f"{prefix}_summary.json"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _episode_len(summary: dict[str, Any]) -> int:
    ms = summary.get("max_steps")
    if ms is not None:
        return int(ms)
    return 30


def rewards_by_episode(rewards: np.ndarray, episode_len: int) -> np.ndarray:
    """Mean reward per episode (truncate if length not divisible)."""
    n = (len(rewards) // episode_len) * episode_len
    if n == 0:
        return np.array([], dtype=np.float32)
    r = rewards[:n].reshape(-1, episode_len)
    return r.mean(axis=1).astype(np.float32)


def plot_figure5(
    prefix_dqn: str,
    prefix_ddpg: str,
    base: Path,
    out: Path,
    *,
    ema_span: int = 15,
) -> None:
    """Convergence over training episodes (mean reward per episode, smoothed)."""
    s_dqn = _load_summary(prefix_dqn, base)
    s_dd = _load_summary(prefix_ddpg, base)
    ep_len = _episode_len(s_dqn) or _episode_len(s_dd)

    r_dqn = np.load(base / f"{prefix_dqn}_rewards.npy")
    r_ddpg = np.load(base / f"{prefix_ddpg}_rewards.npy")
    e_dqn = rewards_by_episode(r_dqn, ep_len)
    e_dd = rewards_by_episode(r_ddpg, ep_len)
    n = min(len(e_dqn), len(e_dd))
    e_dqn, e_dd = e_dqn[:n], e_dd[:n]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(n)
    ax.plot(x, e_dqn, alpha=0.35, color="tab:blue", linewidth=0.8, label="DQN (per-episode mean)")
    ax.plot(x, e_dd, alpha=0.35, color="tab:orange", linewidth=0.8, label="DDPG (per-episode mean)")
    if n >= ema_span:
        ax.plot(x, ema_filter(e_dqn, ema_span), color="tab:blue", lw=2.0, label=f"DQN EMA({ema_span})")
        ax.plot(x, ema_filter(e_dd, ema_span), color="tab:orange", lw=2.0, label=f"DDPG EMA({ema_span})")
    ax.set_xlabel("Training episode index")
    ax.set_ylabel(r"Mean reward per episode ($\approx$ episode length %d steps)" % ep_len)
    ax.set_title("Figure 5 — Convergence of the DRL algorithm over training episodes")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    add_partition_card(fig, base, prefix_dqn, prefix_ddpg, loc="upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure6(
    prefixes: list[str],
    labels: list[str],
    base: Path,
    out: Path,
    *,
    card_prefix_dqn: str | None = None,
    card_prefix_ddpg: str | None = None,
) -> None:
    """Cumulative rewards over separated simulations (same or different seeds)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(len(prefixes), 1)))
    for i, (pref, lab) in enumerate(zip(prefixes, labels)):
        r = np.load(base / f"{pref}_rewards.npy")
        cum = np.cumsum(r.astype(np.float64))
        ax.plot(np.arange(len(cum)), cum, color=cmap[i % 10], lw=1.6, label=lab)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Figure 6 — Cumulative rewards over separated simulations")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    if card_prefix_dqn and card_prefix_ddpg:
        add_partition_card(fig, base, card_prefix_dqn, card_prefix_ddpg, loc="upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure7(
    prefix_dqn: str,
    prefix_ddpg: str,
    base: Path,
    out: Path,
    *,
    rolling: int = 200,
    show_partition_card: bool = True,
) -> None:
    """Achievable user rates and ergodic (rolling-mean) sum rate: DQN vs DDPG."""
    r1_d = np.load(base / f"{prefix_dqn}_r1.npy")
    r2_d = np.load(base / f"{prefix_dqn}_r2.npy")
    rs_d = np.load(base / f"{prefix_dqn}_rsum.npy")
    r1_g = np.load(base / f"{prefix_ddpg}_r1.npy")
    r2_g = np.load(base / f"{prefix_ddpg}_r2.npy")
    rs_g = np.load(base / f"{prefix_ddpg}_rsum.npy")

    n = min(len(r1_d), len(r1_g))
    r1_d, r2_d, rs_d = r1_d[:n], r2_d[:n], rs_d[:n]
    r1_g, r2_g, rs_g = r1_g[:n], r2_g[:n], rs_g[:n]

    def roll(a: np.ndarray) -> np.ndarray:
        return ema_filter(a, rolling) if rolling >= 2 else a

    x = np.arange(n)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    axes[0, 0].plot(x, r1_d, alpha=0.2, color="tab:blue")
    axes[0, 0].plot(x, r1_g, alpha=0.2, color="tab:orange")
    axes[0, 0].plot(x, roll(r1_d), color="tab:blue", lw=1.5, label="DQN")
    axes[0, 0].plot(x, roll(r1_g), color="tab:orange", lw=1.5, label="DDPG")
    axes[0, 0].set_ylabel(r"$R_n$ (near user)")
    axes[0, 0].set_title("Achievable near-user rate")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, r2_d, alpha=0.2, color="tab:blue")
    axes[0, 1].plot(x, r2_g, alpha=0.2, color="tab:orange")
    axes[0, 1].plot(x, roll(r2_d), color="tab:blue", lw=1.5, label="DQN")
    axes[0, 1].plot(x, roll(r2_g), color="tab:orange", lw=1.5, label="DDPG")
    axes[0, 1].set_ylabel(r"$R_f$ (far user)")
    axes[0, 1].set_title("Achievable far-user rate")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, rs_d, alpha=0.2, color="tab:blue")
    axes[1, 0].plot(x, rs_g, alpha=0.2, color="tab:orange")
    axes[1, 0].plot(x, roll(rs_d), color="tab:blue", lw=1.5, label="DQN")
    axes[1, 0].plot(x, roll(rs_g), color="tab:orange", lw=1.5, label="DDPG")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 0].set_ylabel(r"$R_n + R_f$")
    axes[1, 0].set_title("Instantaneous sum rate (raw + smoothed)")
    axes[1, 0].grid(True, alpha=0.25)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(x, roll(rs_d), color="tab:blue", lw=1.8, label="DQN (ergodic approx.)")
    axes[1, 1].plot(x, roll(rs_g), color="tab:orange", lw=1.8, label="DDPG (ergodic approx.)")
    axes[1, 1].set_xlabel("Training step")
    axes[1, 1].set_ylabel(r"Smoothed $R_n + R_f$")
    axes[1, 1].set_title(rf"Ergodic-style sum rate (EMA span {rolling})")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Figure 7 — User rates and sum rate: DQN vs DDPG", y=1.02, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    if show_partition_card:
        add_partition_card(fig, base, prefix_dqn, prefix_ddpg, loc="upper right", fontsize=7.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure8(
    prefix_dqn: str,
    prefix_ddpg: str,
    base: Path,
    out: Path,
    *,
    rolling: int = 200,
) -> None:
    """Hybrid ISAC (trained) vs passive NOMA (all elements for comm) on same trajectories."""
    p_pd = base / f"{prefix_dqn}_rsum_passive.npy"
    p_pg = base / f"{prefix_ddpg}_rsum_passive.npy"
    if not p_pd.is_file() or not p_pg.is_file():
        raise FileNotFoundError(f"Need both {p_pd.name} and {p_pg.name}")

    rs_h_d = np.load(base / f"{prefix_dqn}_rsum.npy")
    rs_p_d = np.load(p_pd)
    rs_h_g = np.load(base / f"{prefix_ddpg}_rsum.npy")
    rs_p_g = np.load(p_pg)

    n = min(len(rs_h_d), len(rs_h_g))
    rs_h_d, rs_p_d = rs_h_d[:n], rs_p_d[:n]
    rs_h_g, rs_p_g = rs_h_g[:n], rs_p_g[:n]

    def roll(a: np.ndarray) -> np.ndarray:
        return ema_filter(a, rolling) if rolling >= 2 else a

    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, roll(rs_h_d), color="tab:blue", lw=1.6, label="DQN — hybrid ISAC $R_n+R_f$")
    ax.plot(
        x,
        roll(rs_p_d),
        color="tab:blue",
        ls="--",
        lw=1.4,
        alpha=0.9,
        label=r"DQN — passive NOMA (comm-only split, $L_t=0$)",
    )
    ax.plot(x, roll(rs_h_g), color="tab:orange", lw=1.6, label="DDPG — hybrid ISAC $R_n+R_f$")
    ax.plot(x, roll(rs_p_g), color="tab:orange", ls="--", lw=1.4, alpha=0.9, label="DDPG — passive NOMA")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Smoothed total rate (bits/s/Hz)")
    ax.set_title("Figure 8 — Hybrid ISAC vs entirely passive NOMA (comm-only RIS split)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    add_partition_card(fig, base, prefix_dqn, prefix_ddpg, loc="upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure9_losses(
    prefix_dqn: str,
    prefix_ddpg: str,
    base: Path,
    out: Path,
    *,
    rolling: int = 50,
    show_partition_card: bool = True,
) -> None:
    """DQN MSE loss vs DDPG critic loss (diagnostic)."""
    l_d = np.load(base / f"{prefix_dqn}_losses.npy")
    l_g = np.load(base / f"{prefix_ddpg}_losses.npy")
    n = min(len(l_d), len(l_g))
    l_d, l_g = l_d[:n], l_g[:n]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.plot(x, l_d, alpha=0.15, color="tab:blue")
    ax.plot(x, l_g, alpha=0.15, color="tab:orange")
    ax.plot(x, ema_filter(l_d, rolling), color="tab:blue", lw=1.4, label="DQN loss EMA")
    ax.plot(x, ema_filter(l_g, rolling), color="tab:orange", lw=1.4, label="DDPG critic loss EMA")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss (diagnostic)")
    if float(np.max(l_d)) > 0 and float(np.max(l_g)) > 0:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    if show_partition_card:
        add_partition_card(fig, base, prefix_dqn, prefix_ddpg, loc="upper right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure10_partitions_jfi(
    prefix_dqn: str,
    prefix_ddpg: str,
    base: Path,
    out: Path,
) -> None:
    """RIS partition trajectories and JFI — useful for explaining policy behavior."""
    p_d = np.load(base / f"{prefix_dqn}_partitions.npy")
    j_d = np.load(base / f"{prefix_dqn}_jfi.npy")
    p_g = np.load(base / f"{prefix_ddpg}_partitions.npy")
    j_g = np.load(base / f"{prefix_ddpg}_jfi.npy")
    n = min(len(p_d), len(p_g))
    p_d, j_d = p_d[:n], j_d[:n]
    p_g, j_g = p_g[:n], j_g[:n]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5))
    for ax, p, title in [
        (axes[0, 0], p_d, "DQN"),
        (axes[0, 1], p_g, "DDPG"),
    ]:
        ax.stackplot(x, p[:, 0], p[:, 1], p[:, 2], labels=[r"$a_n$", r"$a_f$", r"$a_t$"], alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Partition share")
        ax.set_title(f"{title} — RIS partition mix over training")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)

    for ax, j, title in [
        (axes[1, 0], j_d, "DQN"),
        (axes[1, 1], j_g, "DDPG"),
    ]:
        ax.plot(x, j, color="tab:green", lw=0.8, alpha=0.7)
        ax.plot(x, ema_filter(j, 100), color="darkgreen", lw=1.5, label="EMA(100)")
        ax.set_ylim(0.49, 1.02)
        ax.set_xlabel("Training step")
        ax.set_ylabel(r"JFI$(R_n,R_f)$")
        ax.set_title(f"{title} — fairness index")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Extra — Partition dynamics and Jain fairness", y=1.01, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.76, 0.99])
    add_partition_card(fig, base, prefix_dqn, prefix_ddpg, loc="upper right", fontsize=7.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_figure_sweep(
    sweep_json: Path,
    out: Path,
) -> None:
    """Heatmap or line plot from ``paper_sweep.py`` output."""
    with open(sweep_json, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", [])
    if not rows:
        return

    Ls = sorted({int(r["L"]) for r in rows})
    steps_list = sorted({int(r["steps"]) for r in rows})
    mat_dqn = np.full((len(Ls), len(steps_list)), np.nan, dtype=np.float64)
    mat_ddpg = np.full_like(mat_dqn, np.nan)

    for r in rows:
        i = Ls.index(int(r["L"]))
        j = steps_list.index(int(r["steps"]))
        mat_dqn[i, j] = float(r.get("rsum_mean_dqn", np.nan))
        mat_ddpg[i, j] = float(r.get("rsum_mean_ddpg", np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, mat, title in [
        (axes[0], mat_dqn, "DQN — tail mean $R_n+R_f$"),
        (axes[1], mat_ddpg, "DDPG — tail mean $R_n+R_f$"),
    ]:
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(np.arange(len(steps_list)))
        ax.set_xticklabels([str(s) for s in steps_list])
        ax.set_yticks(np.arange(len(Ls)))
        ax.set_yticklabels([str(x) for x in Ls])
        ax.set_xlabel("Training steps")
        ax.set_ylabel(r"Number of reflective elements $L$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Figure — Sum rate vs $L$ and training length (tail window)", y=1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    default_dir = RESULTS_DIR / "figures"
    parser = argparse.ArgumentParser(description="Paper figures from results/*.npy")
    parser.add_argument("--results-dir", type=str, default="", help="Override results directory.")
    parser.add_argument("--out-dir", type=str, default=str(default_dir), help="Figure output directory.")
    parser.add_argument(
        "--prefix-dqn",
        type=str,
        default=DEFAULT_LOG_PREFIX_DQN,
    )
    parser.add_argument(
        "--prefix-ddpg",
        type=str,
        default=DEFAULT_LOG_PREFIX_DDPG,
    )
    parser.add_argument("--rolling", type=int, default=200, help="EMA span for ergodic-style curves.")
    parser.add_argument(
        "--fig6-prefixes",
        type=str,
        default="",
        help="Comma-separated log prefixes for Fig.6 (default: DQN+DDPG only).",
    )
    parser.add_argument(
        "--fig6-labels",
        type=str,
        default="",
        help="Comma-separated labels matching --fig6-prefixes.",
    )
    parser.add_argument(
        "--sweep-json",
        type=str,
        default="",
        help="Optional path to sweep_results.json from paper_sweep.py.",
    )
    parser.add_argument(
        "--skip-sweep-fig",
        action="store_true",
        help="Do not plot sweep heatmap.",
    )
    args = parser.parse_args()

    base = Path(args.results_dir) if args.results_dir.strip() else RESULTS_DIR
    out_dir = Path(args.out_dir)
    ensure_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_figure5(args.prefix_dqn, args.prefix_ddpg, base, out_dir / "figure5_convergence_episodes.png")
    # Fig 6: default two curves; user can pass more prefixes for multi-seed runs
    if args.fig6_prefixes.strip():
        prefs = [p.strip() for p in args.fig6_prefixes.split(",") if p.strip()]
        if args.fig6_labels.strip():
            labs = [x.strip() for x in args.fig6_labels.split(",")]
            while len(labs) < len(prefs):
                labs.append(labs[-1] if labs else "run")
        else:
            labs = prefs
    else:
        prefs = [args.prefix_dqn, args.prefix_ddpg]
        labs = ["DQN", "DDPG"]
    plot_figure6(
        prefs,
        labs,
        base,
        out_dir / "figure6_cumulative_rewards.png",
        card_prefix_dqn=args.prefix_dqn,
        card_prefix_ddpg=args.prefix_ddpg,
    )

    plot_figure7(
        args.prefix_dqn,
        args.prefix_ddpg,
        base,
        out_dir / "figure7_rates_dqn_ddpg.png",
        rolling=args.rolling,
    )

    rpd = base / f"{args.prefix_dqn}_rsum_passive.npy"
    rpg = base / f"{args.prefix_ddpg}_rsum_passive.npy"
    if rpd.is_file() and rpg.is_file():
        try:
            plot_figure8(
                args.prefix_dqn,
                args.prefix_ddpg,
                base,
                out_dir / "figure8_hybrid_vs_passive.png",
                rolling=args.rolling,
            )
        except OSError as e:
            print(f"Skip Figure 8: {e}")
    else:
        print(
            f"Skip Figure 8: need both {rpd.name} and {rpg.name} "
            f"(train DQN and DDPG after updating the env/logging)."
        )

    plot_figure9_losses(
        args.prefix_dqn,
        args.prefix_ddpg,
        base,
        out_dir / "figure9_losses.png",
        rolling=max(20, args.rolling // 4),
    )

    plot_figure10_partitions_jfi(
        args.prefix_dqn,
        args.prefix_ddpg,
        base,
        out_dir / "figure10_partitions_jfi.png",
    )

    sj = args.sweep_json.strip()
    if not args.skip_sweep_fig and sj:
        p = Path(sj)
        if not p.is_absolute():
            p = base / p.name
        if p.is_file():
            plot_figure_sweep(p, out_dir / "figure_sweep_L_steps.png")
        else:
            print(f"No sweep JSON at {p}; skip sweep figure.")

    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
