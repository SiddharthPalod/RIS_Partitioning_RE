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
        "--lambda1",
        type=float,
        default=0.6,
        help="lambda_1: weight on JFI(R_n, R_f).",
    )
    parser.add_argument(
        "--lambda2",
        type=float,
        default=0.008,
        help="lambda_2: weight on ASIR (raw).",
    )
    parser.add_argument(
        "--ddpg-entropy",
        type=float,
        default=0.06,
        help="DDPG only: entropy bonus weight on softmax policy (reduces simplex collapse).",
    )
    parser.add_argument(
        "--ddpg-noise-start",
        type=float,
        default=0.22,
        help="DDPG only: Dirichlet mix weight at step 0 (exploration).",
    )
    parser.add_argument(
        "--ddpg-noise-end",
        type=float,
        default=0.10,
        help="DDPG only: Dirichlet mix weight at last step (decays linearly).",
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

    env_kwargs: dict = {"lambda_1": args.lambda1, "lambda_2": args.lambda2}
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
