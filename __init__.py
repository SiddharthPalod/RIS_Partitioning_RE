"""
Paper5: ISAC-RIS DRL (DQN + DDPG).

Package layout (information architecture)
-----------------------------------------
``config``   Paths and run prefixes (``results/``).
``env``      Gym-style environment (partitions, reward).
``agents``   Neural nets: DQN, DDPG actor/critic, replay buffer.
``training`` Training loops and saving ``.npy`` / ``_summary.json``.

Runnable entry points at the project root: ``main.py`` (train), ``compare.py``, ``visual.py``.

From the ``Paper5`` directory::

    python main.py --algo dqn
    python compare.py
    python visual.py --algo dqn
"""

__version__ = "0.1.0"
