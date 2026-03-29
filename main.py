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
        default=0.025,
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
        default=0.06,
        help="DDPG only: Dirichlet mix weight at last step (decays linearly).",
    )
    args = parser.parse_args()

    env_kwargs = {"lambda_1": args.lambda1, "lambda_2": args.lambda2}

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
