"""Train DQN or DDPG."""

import argparse

from training.loops import run_ddpg, run_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper5 training runner")
    parser.add_argument(
        "--algo",
        type=str,
        default="dqn",
        choices=["dqn", "ddpg"],
        help="Algorithm to run.",
    )
    parser.add_argument("--steps", type=int, default=8000, help="Total training steps.")
    parser.add_argument(
        "--w-c",
        "--lambda1",
        type=float,
        default=0.55,
        dest="w_c",
        help="w_c: weight on communication term normalize(R_n + R_f).",
    )
    parser.add_argument(
        "--w-r",
        "--lambda2",
        type=float,
        default=0.25,
        dest="w_r",
        help="w_r: weight on sensing term normalize(ASIR).",
    )
    parser.add_argument(
        "--w-f",
        type=float,
        default=0.20,
        dest="w_f",
        help="w_f: weight on communication fairness term JFI(R_n, R_f).",
    )
    parser.add_argument("--w-qos-f", type=float, default=0.0, dest="w_qos_f", help="Penalty weight for far-user QoS shortfall max(0, rf_min - R_f).")
    parser.add_argument("--w-qos-s", type=float, default=0.0, dest="w_qos_s", help="Penalty weight for sensing QoS shortfall max(0, asir_min - ASIR).")
    parser.add_argument("--rf-min", type=float, default=0.0, dest="rf_min", help="Minimum far-user rate target.")
    parser.add_argument("--asir-min", type=float, default=0.0, dest="asir_min", help="Minimum sensing ASIR target.")
    parser.add_argument(
        "--max-rsum",
        type=float,
        default=1.2e-3,
        dest="max_rsum",
        help="Normalization scale for (R_n + R_f).",
    )
    parser.add_argument(
        "--max-asir",
        type=float,
        default=8.5e-4,
        dest="max_asir",
        help="Normalization scale for ASIR.",
    )
    parser.add_argument(
        "--ddpg-entropy",
        type=float,
        default=0.01,
        help="DDPG only: entropy bonus weight on softmax policy (reduces simplex collapse).",
    )
    parser.add_argument(
        "--ddpg-noise-start",
        type=float,
        default=0.12,
        help="DDPG only: Dirichlet mix weight at step 0 (exploration).",
    )
    parser.add_argument(
        "--ddpg-noise-end",
        type=float,
        default=0.02,
        help="DDPG only: Dirichlet mix weight at last step (decays linearly).",
    )
    parser.add_argument(
        "--adaptive-norm",
        action="store_true",
        default=True,
        help="Use running reward normalization scales (recommended).",
    )
    parser.add_argument(
        "--fixed-norm",
        action="store_false",
        dest="adaptive_norm",
        help="Disable adaptive normalization and use fixed max-rsum/max-asir.",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="Number of RIS elements (env). Default: env default.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        dest="max_steps",
        help="Steps per episode (env). Default: env default.",
    )
    parser.add_argument(
        "--pn-dbm",
        type=float,
        default=None,
        dest="pn_dbm",
        help="Near-user transmit power in dBm (env). Default: env default.",
    )
    parser.add_argument(
        "--pt-dbm",
        type=float,
        default=None,
        dest="pt_dbm",
        help="BS sensing transmit power in dBm (env). Default: env default.",
    )
    args = parser.parse_args()

    env_kwargs: dict = {
        "w_c": args.w_c,
        "w_r": args.w_r,
        "w_f": args.w_f,
        "w_qos_f": args.w_qos_f,
        "w_qos_s": args.w_qos_s,
        "rf_min": args.rf_min,
        "asir_min": args.asir_min,
        "max_rsum": args.max_rsum,
        "max_asir": args.max_asir,
        "adaptive_norm": args.adaptive_norm,
    }
    if args.L is not None:
        env_kwargs["L"] = args.L
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    if args.pn_dbm is not None:
        env_kwargs["Pn_dBm"] = args.pn_dbm
    if args.pt_dbm is not None:
        env_kwargs["Pt_dBm"] = args.pt_dbm

    if args.algo == "dqn":
        run_dqn(total_steps=args.steps, env_kwargs=env_kwargs)
    else:
        run_ddpg(
            total_steps=args.steps,
            env_kwargs=env_kwargs,
            entropy_coef=args.ddpg_entropy,
            noise_scale_start=args.ddpg_noise_start,
            noise_scale_end=args.ddpg_noise_end,
        )


if __name__ == "__main__":
    main()
