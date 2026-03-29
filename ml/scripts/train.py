"""Train DQN or DDPG."""

import argparse

from ml.training.loops import run_ddpg, run_dqn


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
    args = parser.parse_args()

    env_kwargs = {"lambda_1": args.lambda1, "lambda_2": args.lambda2}

    if args.algo == "dqn":
        run_dqn(total_steps=args.steps, env_kwargs=env_kwargs)
    else:
        run_ddpg(total_steps=args.steps, env_kwargs=env_kwargs)


if __name__ == "__main__":
    main()
