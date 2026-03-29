# Paper5: ISAC-RIS DRL (DQN + DDPG)

This folder recreates the `Paper4/5` training style for a new scenario with:
- Near communication user
- Far communication user
- Sensing target echo

The implementation supports two training pipelines:
- DQN (discrete delta-step actions)
- DDPG (continuous partition actions)

## Layout (where to work)

See **`docs/STRUCTURE.md`** for a full folder map. In short:

| Area | Package path |
|------|----------------|
| Paths & `results/` | `ml/config/paths.py` |
| Environment & reward | `ml/env/simple_isac_env.py` |
| DQN / DDPG networks | `ml/agents/` |
| Training loops & saving | `ml/training/loops.py` |
| Command-line tools | `ml/scripts/` (`train`, `compare`, `visual`) |

**Run from the `Paper5` directory** (so `import ml` works).

Equivalent ways to train:

```bash
python main.py --algo dqn
python -m ml.scripts.train --algo dqn
```

## Files (documentation)

- `Docs.md`: methodology and design notes

## Action Map

The action index controls how a fixed delta (`0.05`) is shifted:
- `0`: near -> far
- `1`: near -> sensing
- `2`: far -> near
- `3`: far -> sensing
- `4`: sensing -> near
- `5`: sensing -> far

Partitions are clipped to stay above a minimum share and renormalized to sum to 1.

## Reward

\[
r_t = \lambda_1 \, \mathrm{JFI}(R_n, R_f) + \lambda_2 \, \mathrm{ASIR}
\]

- `JFI` is Jain fairness over the two communication rates `(R_n, R_f)`.
- `ASIR` enters **without** extra normalization (tune `lambda_2` for scale).

Defaults (CLI `--lambda1` / `--lambda2`):

- `lambda_1 = 0.6`
- `lambda_2 = 0.008` (roughly comparable magnitude to the old `0.4 * ASIR/50` when ASIR \(\approx 50\))

## Run

From `Paper5`:

```bash
python main.py --algo dqn
python main.py --algo ddpg
# or
python -m ml.scripts.train --algo dqn
```

Optional:

```bash
python main.py --algo ddpg --steps 12000
python -m ml.scripts.train --lambda1 0.6 --lambda2 0.008
```

All outputs go under **`results/`** (auto-created):

**NumPy series** (per algorithm prefix, e.g. `training_dqn` / `training_ddpg`):

- `training_dqn_rewards.npy`, `training_dqn_losses.npy`, …  
- Same stems: `actions`, `jfi`, `asir`, `partitions`, `r1`, `r2`, `rsum`, `ravg`.

**JSON summaries** (written at the end of each training run):

- `training_dqn_summary.json`, `training_ddpg_summary.json` — windowed means/stds and metadata (`lambda_1`, `lambda_2`, steps).

**Compare script** also writes `results/compare.png` (unless `--out -` for interactive only) and `results/compare_summary.json`.

Visualize one run:

```bash
python visual.py --algo dqn
python -m ml.scripts.visual --algo ddpg
```

Compare both with the same seed (trains then plots):

```bash
python compare.py --seed 42 --steps 8000
python -m ml.scripts.compare --seed 42 --no-train
python compare.py --out -   # interactive plot only (no file)
```

## Notes

Current equations are placeholder synthetic mappings for fast RL prototyping.
You can replace channel sampling and metric equations in **`ml/env/simple_isac_env.py`** with your analytical model (SINR/outage/echo formulations) without changing the training loop.

If you still have an old **`analytics/`** folder from a previous revision, move or delete it; all new runs write under **`results/`** only.
