# Paper5 folder map (information architecture)

```text
Paper5/                         # Project root (cwd when you run Python)
├── config/                     # Paths, RESULTS_DIR, log prefixes
├── env/                        # Simulation: SimpleISACRISEnv
├── agents/                     # DQN, DDPG (actor/critic/replay)
├── training/                   # Training loops + npy/json export
├── results/                    # Outputs only (npy, json, compare.png)
├── main.py                     # Train DQN or DDPG
├── compare.py                  # Train both + comparison plot
├── visual.py                   # Plot logs from results/
├── README.md
└── docs/
```

| Goal | Location |
|------|----------|
| Channel model, reward definition, partitions | `env/simple_isac_env.py` |
| Network sizes / DQN / DDPG | `agents/` |
| Training hyperparameters, logging | `training/loops.py` |
| CLI flags for training | `main.py` |

Run from the **`Paper5`** directory (`python main.py`, `python compare.py`, etc.).