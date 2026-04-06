import numpy as np

from env.metrics import jains_fairness_index


class SimpleISACRISEnv:
    """
    ISAC-NOMA environment with three RIS zones:
      - near user communication zone
      - far user communication zone
      - sensing target echo zone

    State:
      [h_n, h_f, h_t, a_n, a_f, a_t]
      where h_* are channel gains and a_* are RIS partitions.

    Action:
      One of 6 delta-step transfers between partitions.

    Reward:
      r_t = lambda_1 * JFI(R_n, R_f) + lambda_2 * ASIR
    """

    def __init__(
        self,
        delta: float = 0.05,
        min_partition: float = 0.05,
        max_steps: int = 30,
        w_c: float = 0.7,
        w_r: float = 0.3,
        # Backward compatibility with old names.
        lambda_1: float | None = None,
        lambda_2: float | None = None,
        max_rsum: float = 150.0,
        max_asir: float = 20.0,
        adaptive_norm: bool = True,
        norm_beta: float = 0.995,
        min_norm_ref_rsum: float = 1e-3,
        min_norm_ref_asir: float = 1e-3,
    ):
        self.state_dim = 6
        self.n_actions = 6
        self.delta = float(delta)
        self.min_partition = float(min_partition)
        self.max_steps = int(max_steps)
        if lambda_1 is not None and w_c == 0.7:
            w_c = lambda_1
        if lambda_2 is not None and w_r == 0.3:
            w_r = lambda_2
        self.w_c = float(w_c)
        self.w_r = float(w_r)
        self.max_rsum = float(max_rsum)
        self.max_asir = float(max_asir)
        self.adaptive_norm = bool(adaptive_norm)
        self.norm_beta = float(norm_beta)
        self.min_norm_ref_rsum = float(min_norm_ref_rsum)
        self.min_norm_ref_asir = float(min_norm_ref_asir)
        self.reset()

    def reset(self):
        self.t = 0
        self.a_n, self.a_f, self.a_t = 0.33, 0.33, 0.34
        self._rsum_ref = max(self.min_norm_ref_rsum, 1e-12)
        self._asir_ref = max(self.min_norm_ref_asir, 1e-12)
        self._sample_channels()
        return self._get_state()

    def _transfer_partition(self, from_attr: str, to_attr: str):
        """Move up to delta from one partition to another without going below min_partition.
        Clamps the donor first, then transfers only the actual amount removed so the
        three partitions stay nonnegative and their sum is unchanged (no drift from
        subtract-then-clip-then-renormalize).
        """
        donor = getattr(self, from_attr)
        recip = getattr(self, to_attr)
        new_donor = max(self.min_partition, donor - self.delta)
        actual = donor - new_donor
        setattr(self, from_attr, new_donor)
        setattr(self, to_attr, recip + actual)

    def _sample_channels(self):
        # Placeholder channel model; replace with analytical channels later.
        self.h_n = float(np.random.uniform(0.6, 1.0))
        self.h_f = float(np.random.uniform(0.2, 0.5))
        self.h_t = float(np.random.uniform(0.4, 0.8))

    def _get_state(self):
        return np.array(
            [self.h_n, self.h_f, self.h_t, self.a_n, self.a_f, self.a_t],
            dtype=np.float32,
        )

    def _apply_action(self, action: int):
        # Delta-step action map (donor clamped before credit to recipient; sum preserved).
        if action == 0:  # near -> far
            self._transfer_partition("a_n", "a_f")
        elif action == 1:  # near -> sensing
            self._transfer_partition("a_n", "a_t")
        elif action == 2:  # far -> near
            self._transfer_partition("a_f", "a_n")
        elif action == 3:  # far -> sensing
            self._transfer_partition("a_f", "a_t")
        elif action == 4:  # sensing -> near
            self._transfer_partition("a_t", "a_n")
        elif action == 5:  # sensing -> far
            self._transfer_partition("a_t", "a_f")

        # Keep every zone active and renormalize (handles float drift).
        parts = np.clip([self.a_n, self.a_f, self.a_t], self.min_partition, 1.0)
        parts = parts / np.sum(parts)
        self.a_n, self.a_f, self.a_t = map(float, parts)

    def _compute_metrics(self):
        # Placeholder equations:
        # Communication: single-hop BS–RIS–user (same scaling for near/far).
        r_n = 10.0 * np.log2(1.0 + 100.0 * self.a_n * self.h_n)
        r_f = 10.0 * np.log2(1.0 + 100.0 * self.a_f * self.h_f)
        # Sensing echo: BS–RIS–target–RIS–BS — double attenuation vs comm; use a much
        # smaller effective SNR than 100*a*h and h_t^2 for round-trip path loss.
        sense_linear = 4.0 * self.a_t * (self.h_t**2)
        asir = 15.0 * np.log2(1.0 + sense_linear)

        # Jain fairness over communication users
        jfi = jains_fairness_index([r_n, r_f])
        rsum = float(r_n + r_f)
        if self.adaptive_norm:
            b = np.clip(self.norm_beta, 0.0, 0.9999)
            self._rsum_ref = max(
                self.min_norm_ref_rsum,
                float(b * self._rsum_ref),
                float(abs(rsum)),
            )
            self._asir_ref = max(
                self.min_norm_ref_asir,
                float(b * self._asir_ref),
                float(abs(float(asir))),
            )
            denom_rsum = self._rsum_ref
            denom_asir = self._asir_ref
        else:
            denom_rsum = max(self.max_rsum, 1e-8)
            denom_asir = max(self.max_asir, 1e-8)

        rsum_norm = float(np.clip(rsum / denom_rsum, 0.0, 1.0))
        asir_norm = float(np.clip(float(asir) / denom_asir, 0.0, 1.0))
        reward = self.w_c * rsum_norm + self.w_r * asir_norm
        return float(r_n), float(r_f), float(asir), float(jfi), float(reward), rsum_norm, asir_norm

    def step(self, action: int):
        self.t += 1
        action = int(np.clip(action, 0, self.n_actions - 1))
        self._apply_action(action)

        # Slowly-varying channels across steps.
        self._sample_channels()

        r_n, r_f, asir, jfi, reward, rsum_norm, asir_norm = self._compute_metrics()
        done = self.t >= self.max_steps
        next_state = self._get_state()
        info = {
            "r_n": r_n,
            "r_f": r_f,
            "asir": asir,
            "jfi": jfi,
            "rsum_norm": rsum_norm,
            "asir_norm": asir_norm,
            "a_n": self.a_n,
            "a_f": self.a_f,
            "a_t": self.a_t,
        }
        return next_state, reward, done, info

    def step_continuous(self, action):
        """
        DDPG-friendly step where action is a 3-element partition proposal:
        [a_n, a_f, a_t]. Values are clipped and renormalized to sum to 1.
        """
        self.t += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError("Continuous action must have 3 values: [a_n, a_f, a_t].")

        parts = np.clip(action, self.min_partition, 1.0)
        parts = parts / np.sum(parts)
        self.a_n, self.a_f, self.a_t = map(float, parts)

        self._sample_channels()

        r_n, r_f, asir, jfi, reward, rsum_norm, asir_norm = self._compute_metrics()
        done = self.t >= self.max_steps
        next_state = self._get_state()
        info = {
            "r_n": r_n,
            "r_f": r_f,
            "asir": asir,
            "jfi": jfi,
            "rsum_norm": rsum_norm,
            "asir_norm": asir_norm,
            "a_n": self.a_n,
            "a_f": self.a_f,
            "a_t": self.a_t,
        }
        return next_state, reward, done, info
