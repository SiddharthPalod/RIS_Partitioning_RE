"""Plot training logs from ``results/``."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from config.paths import RESULTS_DIR


def moving_average(x: np.ndarray, window: int = 100) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Paper5 training logs")
    parser.add_argument(
        "--prefix",
        type=str,
        default="training_dqn",
        help="Filename prefix under results/, e.g. training_dqn -> results/training_dqn_rewards.npy",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="",
        choices=["", "dqn", "ddpg"],
        help="If set, overrides --prefix with training_dqn or training_ddpg.",
    )
    args = parser.parse_args()
    prefix = "training_" + args.algo if args.algo else args.prefix

    base = RESULTS_DIR
    rewards = np.load(base / f"{prefix}_rewards.npy")
    losses = np.load(base / f"{prefix}_losses.npy")
    actions = np.load(base / f"{prefix}_actions.npy")
    jfi = np.load(base / f"{prefix}_jfi.npy")
    asir = np.load(base / f"{prefix}_asir.npy")
    partitions = np.load(base / f"{prefix}_partitions.npy")  # [steps, 3]
    r1 = np.load(base / f"{prefix}_r1.npy")
    r2 = np.load(base / f"{prefix}_r2.npy")
    rsum = np.load(base / f"{prefix}_rsum.npy")

    # Partition-weighted "sum rate": a_n R_n + a_f R_f + a_t ASIR
    a_n, a_f, a_t = partitions[:, 0], partitions[:, 1], partitions[:, 2]
    w_rn = a_n * r1
    w_rf = a_f * r2
    w_asir = a_t * asir
    wsum = w_rn + w_rf + w_asir

    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    fig.suptitle(f"Paper5 logs: {prefix}", fontsize=11)
    ax = axes.ravel()

    ax[0].plot(rewards, alpha=0.4, label="reward")
    ma_r = moving_average(rewards, window=100)
    ax[0].plot(np.arange(len(ma_r)) + (len(rewards) - len(ma_r)), ma_r, label="reward MA(100)")
    ax[0].set_title("Reward")
    ax[0].legend()

    ax[1].plot(losses, color="tab:red")
    ax[1].set_title("Loss")

    if actions.ndim == 1:
        ax[2].plot(actions, color="tab:purple")
        ax[2].set_title("Action Index (0-5)")
    else:
        ax[2].plot(actions[:, 0], label="action_n")
        ax[2].plot(actions[:, 1], label="action_f")
        ax[2].plot(actions[:, 2], label="action_t")
        ax[2].set_title("Continuous Action (DDPG)")
        ax[2].legend()

    ax[3].plot(jfi, color="tab:green", label="JFI")
    # N=2 comm users: Jain index in [1/2, 1], not [0, 1]
    ax[3].set_ylim(0.49, 1.02)
    ax[3].set_title(r"JFI$(R_n,R_f)$ — Jain, $J\in[1/2,1]$ for $N=2$")
    ax[3].legend()

    ax[4].plot(asir, color="tab:orange")
    ax[4].set_title("Sensing ASIR")

    ax[5].plot(partitions[:, 0], label="a_n")
    ax[5].plot(partitions[:, 1], label="a_f")
    ax[5].plot(partitions[:, 2], label="a_t")
    ax[5].set_title("RIS Partitions")
    ax[5].set_ylim(0.0, 1.0)
    ax[5].legend()

    ax[6].plot(w_rn, alpha=0.7, label=r"$a_n R_n$")
    ax[6].plot(w_rf, alpha=0.7, label=r"$a_f R_f$")
    ax[6].plot(w_asir, alpha=0.7, label=r"$a_t \mathrm{ASIR}$")
    ax[6].plot(wsum, color="black", linewidth=1.2, label=r"$a_n R_n + a_f R_f + a_t \mathrm{ASIR}$")
    ax[6].set_title("Partition-weighted rates")
    ax[6].legend(loc="best", fontsize=8)

    ax[7].plot(rsum, alpha=0.5, label=r"$R_n + R_f$ (unweighted)")
    ax[7].plot(wsum, alpha=0.9, label=r"$a_n R_n + a_f R_f + a_t \mathrm{ASIR}$")
    ax[7].set_title("Sum rate comparison")
    ax[7].legend(loc="best", fontsize=8)

    for a in ax:
        a.grid(True, alpha=0.2)
        a.set_xlabel("Step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
