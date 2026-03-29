"""Project root and ``results/`` output directory."""

from pathlib import Path

# Paper5/config/paths.py → parents: config → Paper5 (cwd / project root)
PAPER5_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PAPER5_ROOT / "results"

DEFAULT_LOG_PREFIX_DQN = "training_dqn"
DEFAULT_LOG_PREFIX_DDPG = "training_ddpg"


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR
