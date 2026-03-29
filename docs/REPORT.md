# Paper5 — Implementation report (current)

## Scope

Everything below refers only to the **Paper5** folder: the **`ml/`** package (`config`, `env`, `agents`, `training`, `scripts`), root launchers (`main.py`, `compare.py`, `visual.py`), and `README.md`. See **`docs/STRUCTURE.md`** for paths. ECE / analytical \(R_n, R_f\), ASIR are still **placeholders** until you plug in your real channel and sensing model.

---

## 1. What the code implements now

### Environment (`ml/env/simple_isac_env.py`, class `SimpleISACRISEnv`)

- **State (6D):** `[h_n, h_f, h_t, a_n, a_f, a_t]`.
- **Discrete actions (DQN):** six \(\Delta\)-transfers (default \(\Delta=0.05\)), clip each zone to `[min_partition, 1]`, renormalize to the simplex.
- **Continuous actions (DDPG):** direct `[a_n, a_f, a_t]`, same clip + renormalize.
- **Metrics (synthetic):** log-like rates for near/far and an ASIR-like term; **not** from ECE yet.

### Reward (current)

\[
r_t = \lambda_1 \, \mathrm{JFI}(R_n, R_f) + \lambda_2 \, \mathrm{ASIR}
\]

- **JFI** is Jain fairness on **two communication rates** \((R_n, R_f)\) only.
- **ASIR** is used **in raw form** (no division by `asir_max`, no clipping to \([0,1]\)).
- Defaults in code: **`lambda_1 = 0.6`**, **`lambda_2 = 0.008`** (tunable; README notes that \(\lambda_2 \approx 0.008\) is in the ballpark of the old `0.4 × (ASIR/50)` when ASIR \(\approx 50\)).

### Dynamics

- After each step, **channels are resampled** (`_sample_channels()`), so \((h_n,h_f,h_t)\) change **every timestep** while partitions evolve within an episode (until `max_steps`).

### Training (`ml/training/loops.py`, CLI `ml/scripts/train.py` or root `main.py`)

- **DQN:** MLP Q-network, ε-greedy, replay, periodic hard target sync.
- **DDPG:** softmax actor on the 3-way partition, critic on `[state ∥ action]`, replay, soft target updates (`tau`), Dirichlet exploration noise on the simplex.
- **CLI:** `--algo dqn|ddpg`, `--steps`, **`--lambda1`**, **`--lambda2`** (passed into the env).

### Logging (`ml/config/paths.py`, `results/`)

- All `.npy` series and **`{prefix}_summary.json`** files are written under **`Paper5/results/`** (created automatically).
- Runs do not overwrite each other across algorithms:
  - DQN → prefix **`training_dqn`** → e.g. `results/training_dqn_rewards.npy`, `results/training_dqn_summary.json`, …
  - DDPG → prefix **`training_ddpg`** → same pattern.

### Tools

- **`compare.py`:** same **`--seed`** (default 42), trains DQN then DDPG with the same \(\lambda_1,\lambda_2\), then plots **reward**, **JFI**, and **sum rate \(R_n+R_f\)**. Saves **`results/compare.png`** and **`results/compare_summary.json`** by default; `--no-train` only loads existing `.npy` files; `--out -` opens an interactive plot only.
- **`visual.py`:** **`--prefix`** or **`--algo dqn|ddpg`** to choose which prefixed logs under **`results/`** to plot.

---

## 2. Alignment with a “paper-style” optimization story

| Goal (as described in discussion) | In code now |
|-----------------------------------|-------------|
| Emphasize fairness between comm users | **Yes:** JFI on \((R_n,R_f)\). |
| Include sensing | **Yes:** linear in **ASIR** with \(\lambda_2\). |
| “Max sum \(R_n + R_f + \mathrm{ASIR}\)” as primary objective | **No:** objective is **JFI + weighted ASIR**, not explicit sum of rates + ASIR. **Sum rate** is logged (`*_rsum.npy`) but **not** the scalar reward. |
| Fairness on **partitions** \(a_n,a_f\) | **No:** JFI is on **rates**, not on \((a_n,a_f)\). |

The **implemented** problem is: maximize (in expectation) \(\lambda_1 \mathrm{JFI} + \lambda_2 \mathrm{ASIR}\) under placeholder physics. That matches the chosen scalar reward; it is **not** identical to a multi-objective “max sum-rate subject to JFI constraint” unless you change the reward or add constraints.

---

## 3. Why DDPG might still not “win” on every metric

- **Different objectives:** High **reward** can trade off **JFI** vs **ASIR**; **sum rate** can move differently from both. After the reward update, **reward scale** can be dominated by **ASIR** if \(\lambda_2 \cdot \mathrm{ASIR}\) is large relative to \(\lambda_1 \cdot \mathrm{JFI}\) (JFI \(\in [0,1]\), ASIR in the tens in the toy model).
- **Non-stationarity:** i.i.d. new channels **every step** makes the MDP hard for both methods; comparisons are **high-variance** unless you fix seeds and use **`compare.py`**.
- **Algorithm / hyperparameters:** Discrete \(\Delta\)-steps vs full simplex, different learning rates and exploration, no Double-DQN, etc., all affect rankings.

**Reproducibility:** `main.py` still does **not** set RNG seeds by default; use **`compare.py --seed`** (or set seeds yourself) for fair DQN vs DDPG runs.

---

## 4. Potential issues and practical fixes

1. **Scale of ASIR vs JFI:** Raw ASIR can make the **critic’s target** large and noisy. Mitigations: reduce **`lambda_2`**, normalize ASIR in the **reward only** while keeping raw ASIR in logs, or standardize rewards in the replay buffer.
2. **Placeholder metrics:** Until ECE is in, conclusions about “which algorithm is better for ISAC” are **toy-only**.
3. **Channel model:** If the paper assumes **block fading** or slow variation, resampling every step is a **deliberate simplification**; consider resampling only on **episode reset** if you want closer alignment.
4. **Fair paper comparison:** Use **`compare.py`** with fixed **`--seed`**, same **`--steps`**, same **`--lambda1` / `--lambda2`**, and report **reward**, **JFI**, and **rsum** separately.

---

## 5. Empirical evaluation (how to report results)

- Artifacts live under **`results/`**: **per-algorithm** `.npy` series, **`_summary.json`** per run, and **`compare_summary.json`** after **`compare.py`**.
- Standard recipe:
  1. `python compare.py --seed 42 --steps 8000` (writes **`results/compare.png`** by default)
  2. Use **`training_dqn_summary.json`** / **`training_ddpg_summary.json`**, or load **`results/training_dqn_*.npy`** and **`results/training_ddpg_*.npy`** for custom stats.

---

## 6. Summary

The **Paper5** codebase implements \(r_t = \lambda_1 \mathrm{JFI} + \lambda_2 \mathrm{ASIR}\), writes all training outputs to **`results/`** (including JSON summaries) with **separate prefixes** for DQN and DDPG, and includes **`compare.py`** for **seeded, side-by-side** training and plotting. Remaining gaps are **placeholder physics**, **no explicit sum-rate term in \(r_t\)**, **rate-based (not partition-based) JFI**, and **fast channel resampling**—tighten these when ECE and the paper’s channel assumptions are integrated.
