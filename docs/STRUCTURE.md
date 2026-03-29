# Paper5 folder map (information architecture)

```text
Paper5/
├── ml/                    # Python package — all importable code
│   ├── config/                # Paths, RESULTS_DIR, log prefixes
│   ├── env/                     # Simulation: SimpleISACRISEnv
│   ├── agents/                  # DQN, DDPG (actor/critic/replay)
│   ├── training/                # Training loops + npy/json export
│   └── scripts/                 # CLIs (train, compare, visual)
├── results/                     # Outputs only (npy, json, compare.png)
├── main.py                      # Thin wrapper → ml.scripts.train
├── compare.py                 # Thin wrapper → ml.scripts.compare
├── visual.py                  # Thin wrapper → ml.scripts.visual
├── README.md
├── REPORT.md
└── ...
```

**What to edit**

| Goal | Location |
|------|----------|
| Channel model, reward definition, partitions | `ml/env/simple_isac_env.py` |
| Network sizes / DQN / DDPG | `ml/agents/` |
| Training hyperparameters, logging | `ml/training/loops.py` |
| CLI flags | `ml/scripts/train.py` (and `compare.py` for compare) |

Run from the **`Paper5`** directory so `import ml` resolves.
