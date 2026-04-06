"""
Sweep over number of reflective elements ``L`` and training ``steps``; save metrics to JSON.

Example (short, for testing):

  python paper_sweep.py --L 256 512 --steps 500 --algo both

Full paper-style runs use larger ``--steps`` and more ``L`` values (can take a long time).
Figures: run ``python paper_figures.py --sweep-json results/sweep_results.json`` after.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config.paths import RESULTS_DIR, ensure_results_dir
from training.loops import run_ddpg, run_dqn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _tail_rsum_mean(prefix: str, base: Path, tail: int = 500) -> float | None:
    p = base / f"{prefix}_rsum.npy"
    if not p.is_file():
        return None
    r = np.load(p)
    n = min(tail, len(r))
    if n == 0:
        return None
    return float(r[-n:].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep L and training steps (DQN / DDPG)")
    parser.add_argument("--L", type=int, nargs="+", required=True, help="RIS element counts.")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="Training steps per run.")
    parser.add_argument(
        "--algo",
        type=str,
        default="both",
        choices=["dqn", "ddpg", "both"],
        help="Which algorithm(s) to run per grid point.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed (same for both algos at a grid point).")
    parser.add_argument("--w-c", "--lambda1", type=float, default=0.55, dest="w_c")
    parser.add_argument("--w-r", "--lambda2", type=float, default=0.25, dest="w_r")
    parser.add_argument("--w-f", type=float, default=0.20, dest="w_f")
    parser.add_argument("--w-qos-f", type=float, default=0.0, dest="w_qos_f")
    parser.add_argument("--w-qos-s", type=float, default=0.0, dest="w_qos_s")
    parser.add_argument("--rf-min", type=float, default=0.0, dest="rf_min")
    parser.add_argument("--asir-min", type=float, default=0.0, dest="asir_min")
    parser.add_argument("--max-rsum", type=float, default=1.2e-3, dest="max_rsum")
    parser.add_argument("--max-asir", type=float, default=8.5e-4, dest="max_asir")
    parser.add_argument(
        "--out-json",
        type=str,
        default="sweep_results.json",
        help="Filename under results/ for JSON summary.",
    )
    args = parser.parse_args()

    ensure_results_dir()
    base = RESULTS_DIR
    out_path = base / args.out_json

    results: list[dict[str, Any]] = []

    for L in args.L:
        for steps in args.steps:
            env_kwargs: dict[str, Any] = {
                "w_c": args.w_c,
                "w_r": args.w_r,
                "w_f": args.w_f,
                "w_qos_f": args.w_qos_f,
                "w_qos_s": args.w_qos_s,
                "rf_min": args.rf_min,
                "asir_min": args.asir_min,
                "max_rsum": args.max_rsum,
                "max_asir": args.max_asir,
                "L": int(L),
            }
            row: dict[str, Any] = {"L": int(L), "steps": int(steps), "seed": args.seed}

            if args.algo in ("dqn", "both"):
                pref_d = f"sweep_L{L}_steps{steps}_dqn"
                set_seed(args.seed)
                run_dqn(total_steps=steps, log_prefix=pref_d, env_kwargs=env_kwargs)
                row["rsum_mean_dqn"] = _tail_rsum_mean(pref_d, base)
                row["prefix_dqn"] = pref_d

            if args.algo in ("ddpg", "both"):
                pref_g = f"sweep_L{L}_steps{steps}_ddpg"
                set_seed(args.seed)
                run_ddpg(total_steps=steps, log_prefix=pref_g, env_kwargs=env_kwargs)
                row["rsum_mean_ddpg"] = _tail_rsum_mean(pref_g, base)
                row["prefix_ddpg"] = pref_g

            results.append(row)
            print(f"Done L={L} steps={steps} -> {row}")

    payload = {
        "w_c": args.w_c,
        "w_r": args.w_r,
        "w_f": args.w_f,
        "w_qos_f": args.w_qos_f,
        "w_qos_s": args.w_qos_s,
        "rf_min": args.rf_min,
        "asir_min": args.asir_min,
        "max_rsum": args.max_rsum,
        "max_asir": args.max_asir,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
