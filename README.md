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
| Paths & `results/` | `config/paths.py` |
| Environment & reward | `env/simple_isac_env.py` |
| DQN / DDPG networks | `agents/` |
| Training loops & saving | `training/loops.py` |
| Entry points | `main.py` (train), `compare.py`, `visual.py` |

**Run from the `Paper5` directory** (project root on Python’s import path).

Train:

```bash
python main.py --algo dqn
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
r_t = w_c \, \widehat{(R_n + R_f)} + w_r \, \widehat{\mathrm{ASIR}}
\]

- Uses explicit communication/sensing trade-off weights (`w_c`, `w_r`).
- Both terms are normalized by reference scales (`max_rsum`, `max_asir`) to prevent one term from dominating due to units/magnitude.

Defaults (CLI `--w-c` / `--w-r`):

- `w_c = 0.7`
- `w_r = 0.3`

## Run

From `Paper5`:

```bash
python main.py --algo dqn
python main.py --algo ddpg
```

Optional:

```bash
python main.py --algo ddpg --steps 12000
python main.py --w-c 0.7 --w-r 0.3 --max-rsum 9e-10 --max-asir 250
```

All outputs go under **`results/`** (auto-created):

**NumPy series** (per algorithm prefix, e.g. `training_dqn` / `training_ddpg`):

- `training_dqn_rewards.npy`, `training_dqn_losses.npy`, …  
- Same stems: `actions`, `jfi`, `asir`, `partitions`, `r1`, `r2`, `rsum`, `ravg`.

**JSON summaries** (written at the end of each training run):

- `training_dqn_summary.json`, `training_ddpg_summary.json` — windowed means/stds and metadata (`w_c`, `w_r`, normalization scales, steps).

**Compare script** also writes `results/compare.png` (unless `--out -` for interactive only) and `results/compare_summary.json`.

Visualize one run:

```bash
python visual.py --algo dqn
python visual.py --algo ddpg
```

Compare both with the same seed (trains then plots):

```bash
python compare.py --seed 42 --steps 8000
python compare.py --seed 42 --steps 8000 --w-c 0.7 --w-r 0.3 --max-rsum 9e-10 --max-asir 250
python compare.py --steps 8000 --seed 42 --w-c 0.35 --w-r 0.20 --w-f 0.15 --w-qos-f 20000 --rf-min 6e-5 --w-qos-s 5000 --asir-min 7.5e-5 --max-rsum 1.2e-3 --max-asir 8.5e-4
python compare.py --seed 42 --no-train
python compare.py --out -   # interactive plot only (no file)
```

## Notes

Current equations are placeholder synthetic mappings for fast RL prototyping.
You can replace channel sampling and metric equations in **`env/simple_isac_env.py`** with your analytical model (SINR/outage/echo formulations) without changing the training loop.

If you still have an old **`analytics/`** folder from a previous revision, move or delete it; all new runs write under **`results/`** only.
