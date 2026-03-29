"""
Paper5: ISAC-RIS DRL (DQN + DDPG).

Package layout (information architecture)
-----------------------------------------
``ml.config``   Paths and run prefixes (``results/``).
``ml.env``      Gym-style environment (partitions, reward).
``ml.agents``   Neural nets: DQN, DDPG actor/critic, replay buffer.
``ml.training`` Training loops and saving ``.npy`` / ``_summary.json``.
``ml.scripts``  Runnable CLIs (train, compare, visual).

From the ``Paper5`` directory::

    python -m ml.scripts.train --algo dqn
    python -m ml.scripts.compare
    python -m ml.scripts.visual --algo dqn

Or use the short launchers in the repo root: ``main.py``, ``compare.py``, ``visual.py``.
"""

__version__ = "0.1.0"
