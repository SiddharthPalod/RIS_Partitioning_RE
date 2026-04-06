import numpy as np

from env.metrics import jains_fairness_index


class ISACRISEnv:
    def __init__(
        self,
        L=1000,
        fc=5.8e9,
        BW=10e6,
        sigma_dBm=-104,
        delta_step=0.05,
        delta_n=0.05,
        delta_f=0.05,
        P_delta=3.0,     # Added: Power back-off for NOMA (dB) 
        Pn_dBm=20.0,
        Pt_dBm=30.0,
        min_partition=0.05,
        w_c=0.55,
        w_r=0.25,
        w_f=0.20,
        w_qos_f=0.0,
        w_qos_s=0.0,
        rf_min=0.0,
        asir_min=0.0,
        # Backward-compatible aliases; used only if explicit w_c/w_r are not passed.
        lambda_1=None,
        lambda_2=None,
        # Reward normalization reference scales (can be tuned from CLI/env kwargs).
        max_rsum=1.2e-3,
        max_asir=8.5e-4,
        adaptive_norm=True,
        norm_beta=0.995,
        min_norm_ref_rsum=1e-12,
        min_norm_ref_asir=1e-6,
        max_steps=30,
    ):
        self.state_dim = 6
        self.n_actions = 6
        self.c = 3e8
        self.fc = fc
        self.lambda_c = self.c / fc
        self.BW = BW
        self.sigma2 = 10 ** ((sigma_dBm - 30) / 10)

        self.delta = delta_step
        self.delta_n = float(delta_n)
        self.delta_f = float(delta_f)
        self.P_delta = float(P_delta) # Added Power back-off
        self.Pn = 10 ** ((float(Pn_dBm) - 30.0) / 10.0)
        self.Pt = 10 ** ((float(Pt_dBm) - 30.0) / 10.0)
        self.min_partition = min_partition
        # Keep compatibility with old kwargs names.
        if lambda_1 is not None and w_c == 0.55:
            w_c = lambda_1
        if lambda_2 is not None and w_r == 0.25:
            w_r = lambda_2
        self.w_c = float(w_c)
        self.w_r = float(w_r)
        self.w_f = float(w_f)
        self.w_qos_f = float(w_qos_f)
        self.w_qos_s = float(w_qos_s)
        self.rf_min = float(rf_min)
        self.asir_min = float(asir_min)
        self.max_rsum = float(max_rsum)
        self.max_asir = float(max_asir)
        self.adaptive_norm = bool(adaptive_norm)
        self.norm_beta = float(norm_beta)
        self.min_norm_ref_rsum = float(min_norm_ref_rsum)
        self.min_norm_ref_asir = float(min_norm_ref_asir)

        self.L = L
        self.max_steps = max_steps

        # Standardized positions
        self.pos_BS = np.array([0, 0])
        self.pos_RIS = np.array([100, 50])
        self.pos_Dn = np.array([598, 0])  # Updated to match paper [cite: 303]
        self.pos_Df = np.array([748, 0])  # Updated to match paper [cite: 303]
        self.pos_T = np.array([148, 100])

        self.RCS = 50
        self.var_tau = 1e-6
        self.w = 0.1
        self.T_dur = 1e-3

        self.reset()

    def reset(self):
        self.t = 0
        third = 1.0 / 3.0
        self.a_n, self.a_f, self.a_t = third, third, third
        # Running references for adaptive normalization.
        self._rsum_ref = max(self.min_norm_ref_rsum, 1e-12)
        self._asir_ref = max(self.min_norm_ref_asir, 1e-12)
        self._generate_channels()
        return self._get_state()

    def _nakagami(self, m, omega, size):
        return np.sqrt(np.random.gamma(m, omega / m, size))

    def _path_loss(self, d):
        alpha = 2.5
        C = (self.lambda_c / (4 * np.pi)) ** 2
        return C * d ** (-alpha)

    def _generate_channels(self):
        m = 3
        d_Dn_R = np.linalg.norm(self.pos_Dn - self.pos_RIS)
        d_Df_R = np.linalg.norm(self.pos_Df - self.pos_RIS)
        d_R_BS = np.linalg.norm(self.pos_RIS - self.pos_BS)
        d_T_R = np.linalg.norm(self.pos_T - self.pos_RIS)

        self.h_DnR = self._nakagami(m, self._path_loss(d_Dn_R), self.L)
        self.h_DfR = self._nakagami(m, self._path_loss(d_Df_R), self.L)
        self.h_RB = self._nakagami(m, self._path_loss(d_R_BS), self.L)
        self.h_TR = self._nakagami(m, self._path_loss(d_T_R), self.L)

    def _sample_channels(self):
        self._generate_channels()

    def _transfer_partition(self, from_attr: str, to_attr: str):
        donor = getattr(self, from_attr)
        recip = getattr(self, to_attr)
        new_donor = max(self.min_partition, donor - self.delta)
        actual = donor - new_donor
        setattr(self, from_attr, new_donor)
        setattr(self, to_attr, recip + actual)

    def _allocate_ris_elements(self):
        """
        Map continuous fractions (a_n, a_f, a_t) to integer element counts that sum to L.
        Uses largest-remainder (Hamilton) so rounding slack is not dumped into Lt
        (independent floors + tail slice would bias sensing vs comm).
        """
        L = self.L
        raw = np.array([self.a_n, self.a_f, self.a_t], dtype=np.float64) * L
        floors = np.floor(raw).astype(np.int64)
        rem = int(L - int(np.sum(floors)))
        fracs = raw - floors.astype(np.float64)
        order = np.argsort(-fracs)
        extras = np.zeros(3, dtype=np.int64)
        for k in range(rem):
            extras[order[k]] += 1
        c = floors + extras
        # At least one element per zone when L allows (valid slicing)
        if L >= 3:
            while c.min() < 1 and c.max() > 1:
                i = int(np.argmax(c))
                j = int(np.argmin(c))
                if c[i] <= 1:
                    break
                c[i] -= 1
                c[j] += 1
        Ln, Lf, Lt = int(c[0]), int(c[1]), int(c[2])
        return Ln, Lf, Lt

    def _allocate_passive_comm_only(self) -> tuple[int, int]:
        """
        Entirely passive NOMA: all L reflective elements serve the two comm users
        (no sensing slice). Split follows a_n : a_f after renormalizing away from a_t.
        """
        L = self.L
        s = float(self.a_n + self.a_f)
        if s <= 1e-12:
            an, af = 0.5, 0.5
        else:
            an = self.a_n / s
            af = self.a_f / s
        raw = np.array([an, af], dtype=np.float64) * L
        floors = np.floor(raw).astype(np.int64)
        rem = int(L - int(np.sum(floors)))
        fracs = raw - floors.astype(np.float64)
        order = np.argsort(-fracs)
        extras = np.zeros(2, dtype=np.int64)
        for k in range(rem):
            extras[order[k]] += 1
        c = floors + extras
        if L >= 2:
            while c.min() < 1 and c.max() > 1:
                i = int(np.argmax(c))
                j = int(np.argmin(c))
                if c[i] <= 1:
                    break
                c[i] -= 1
                c[j] += 1
        return int(c[0]), int(c[1])

    def _rates_from_segments(self, Ln: int, Lf: int, Lt: int) -> tuple[float, float, float]:
        """Return (r_n, r_f, asir) for fixed (Ln, Lf, Lt) on current channel draws."""
        Pn = self.Pn
        Pf = 10 ** (-self.P_delta / 10) * Pn
        Pt = self.Pt

        h_n_sum = np.sum(self.h_DnR[:Ln] * self.h_RB[:Ln])
        h_f_sum = np.sum(self.h_DfR[Ln : Ln + Lf] * self.h_RB[Ln : Ln + Lf])
        # h[-0:] selects the entire array in NumPy; guard Lt==0.
        g_t_sum = (
            np.sum(self.h_TR[-Lt:] * self.h_RB[-Lt:])
            if Lt > 0
            else np.float64(0.0)
        )

        gT_eff = g_t_sum ** 2

        rho = (self.RCS * self.lambda_c ** 2 / (4 * np.pi) ** 3) * (
            2 * np.pi ** 2 / 12
        ) * self.BW ** 2 * self.var_tau

        # Keep interference/power scaling physically consistent between
        # communication and sensing paths. A previous one-sided calibration
        # factor inflated sensing interference and caused policy collapse.
        I_radar = Pt * rho * (np.abs(gT_eff) ** 2)

        gamma_n = (np.abs(h_n_sum) ** 2 * Pn) / (
            np.abs(h_f_sum) ** 2 * Pf + I_radar + self.sigma2
        )
        gamma_f = (np.abs(h_f_sum) ** 2 * Pf) / (
            self.delta_n * np.abs(h_n_sum) ** 2 * Pn + I_radar + self.sigma2
        )
        gamma_bs = (rho * np.abs(gT_eff) ** 2 * Pt) / (
            self.delta_n * np.abs(h_n_sum) ** 2 * Pn
            + self.delta_f * np.abs(h_f_sum) ** 2 * Pf
            + self.sigma2
        )

        rn = float(np.log2(1 + gamma_n))
        rf = float(np.log2(1 + gamma_f))
        asir = float(
            (self.w / (2 * self.T_dur)) * np.log2(1 + 2 * self.T_dur * self.BW * gamma_bs)
        )
        return rn, rf, asir

    def _compute_metrics(self):
        Ln, Lf, Lt = self._allocate_ris_elements()

        rn, rf, asir = self._rates_from_segments(Ln, Lf, Lt)

        Ln_p, Lf_p = self._allocate_passive_comm_only()
        rn_p, rf_p, _ = self._rates_from_segments(Ln_p, Lf_p, 0)

        jfi = jains_fairness_index([rn, rf])
        rsum = rn + rf
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
                float(abs(asir)),
            )
            denom_rsum = self._rsum_ref
            denom_asir = self._asir_ref
        else:
            denom_rsum = max(self.max_rsum, 1e-20)
            denom_asir = max(self.max_asir, 1e-20)

        rsum_norm = float(np.clip(rsum / denom_rsum, 0.0, 1.0))
        asir_norm = float(np.clip(asir / denom_asir, 0.0, 1.0))
        rf_penalty = float(max(0.0, self.rf_min - rf))
        asir_penalty = float(max(0.0, self.asir_min - asir))
        reward = float(
            self.w_c * rsum_norm
            + self.w_r * asir_norm
            + self.w_f * jfi
            - self.w_qos_f * rf_penalty
            - self.w_qos_s * asir_penalty
        )

        return rn, rf, asir, jfi, reward, rn_p, rf_p, rsum_norm, asir_norm, float(denom_rsum), float(denom_asir), rf_penalty, asir_penalty

    def _get_state(self):
        return np.array([
            np.mean(np.abs(self.h_DnR)),
            np.mean(np.abs(self.h_DfR)),
            np.mean(np.abs(self.h_TR)),
            self.a_n, self.a_f, self.a_t
        ], dtype=np.float32)

    def _apply_action(self, action: int):
        if action == 0: self._transfer_partition("a_n", "a_f")
        elif action == 1: self._transfer_partition("a_n", "a_t")
        elif action == 2: self._transfer_partition("a_f", "a_n")
        elif action == 3: self._transfer_partition("a_f", "a_t")
        elif action == 4: self._transfer_partition("a_t", "a_n")
        elif action == 5: self._transfer_partition("a_t", "a_f")

        parts = np.clip([self.a_n, self.a_f, self.a_t], self.min_partition, 1.0)
        parts = parts / np.sum(parts)
        self.a_n, self.a_f, self.a_t = map(float, parts)

    def step(self, action: int):
        self.t += 1
        self._apply_action(int(action))
        self._sample_channels()

        rn, rf, asir, jfi, reward, rn_p, rf_p, rsum_norm, asir_norm, rsum_ref, asir_ref, rf_penalty, asir_penalty = self._compute_metrics()
        rsum_p = rn_p + rf_p

        info = {
            "r_n": rn,
            "r_f": rf,
            "asir": asir,
            "jfi": jfi,
            "r_n_passive": rn_p,
            "r_f_passive": rf_p,
            "rsum_passive": rsum_p,
            "rsum_norm": rsum_norm,
            "asir_norm": asir_norm,
            "rsum_ref": rsum_ref,
            "asir_ref": asir_ref,
            "rf_penalty": rf_penalty,
            "asir_penalty": asir_penalty,
            "a_n": self.a_n,
            "a_f": self.a_f,
            "a_t": self.a_t,
        }
        return self._get_state(), reward, (self.t >= self.max_steps), info

    def step_continuous(self, action):
        self.t += 1
        parts = np.clip(np.atleast_1d(action).flatten(), self.min_partition, 1.0)
        parts = parts / np.sum(parts)
        self.a_n, self.a_f, self.a_t = map(float, parts)

        self._sample_channels()
        rn, rf, asir, jfi, reward, rn_p, rf_p, rsum_norm, asir_norm, rsum_ref, asir_ref, rf_penalty, asir_penalty = self._compute_metrics()
        rsum_p = rn_p + rf_p

        info = {
            "r_n": rn,
            "r_f": rf,
            "asir": asir,
            "jfi": jfi,
            "r_n_passive": rn_p,
            "r_f_passive": rf_p,
            "rsum_passive": rsum_p,
            "rsum_norm": rsum_norm,
            "asir_norm": asir_norm,
            "rsum_ref": rsum_ref,
            "asir_ref": asir_ref,
            "rf_penalty": rf_penalty,
            "asir_penalty": asir_penalty,
            "a_n": self.a_n,
            "a_f": self.a_f,
            "a_t": self.a_t,
        }
        return self._get_state(), reward, (self.t >= self.max_steps), info


# import numpy as np

# class ISACRISEnv:
#     def __init__(
#         self,
#         L=1000,
#         fc=5.8e9,
#         BW=10e6,
#         sigma_dBm=-104,
#         delta_step=0.05,
#         # ImpSIC factors (MATLAB uses delta_n and delta_f). Kept separate from delta_step,
#         # which is the RIS-partition transfer step size for the action.
#         delta_n=0.05,
#         delta_f=0.05,
#         min_partition=0.05,
#         lambda_1=0.6,
#         lambda_2=0.008,
#         max_steps=30,
#     ):
#         self.state_dim = 6
#         self.n_actions = 6
#         self.c = 3e8
#         self.fc = fc
#         self.lambda_c = self.c / fc
#         self.BW = BW
#         self.sigma2 = 10 ** ((sigma_dBm - 30) / 10)

#         self.delta = delta_step
#         self.delta_n = float(delta_n)
#         self.delta_f = float(delta_f)
#         self.min_partition = min_partition
#         self.lambda_1 = lambda_1
#         self.lambda_2 = lambda_2

#         self.L = L
#         self.max_steps = max_steps

#         # Standardized positions for analytical path loss
#         self.pos_BS = np.array([0, 0])
#         self.pos_RIS = np.array([100, 50])
#         self.pos_Dn = np.array([120, 0])
#         self.pos_Df = np.array([400, 0])
#         self.pos_T = np.array([148, 100])

#         self.RCS = 50
#         self.var_tau = 1e-6
#         self.w = 0.1
#         self.T_dur = 1e-3

#         self.reset()

#     def reset(self):
#         self.t = 0
#         self.a_n, self.a_f, self.a_t = 0.33, 0.33, 0.34
#         self._generate_channels()
#         return self._get_state()

#     def _nakagami(self, m, omega, size):
#         # Use real Nakagami magnitudes (coherent RIS combining).
#         # MATLAB's system_model.m generates sqrt(gamrnd(...)) without random phase.
#         return np.sqrt(np.random.gamma(m, omega / m, size))

#     def _path_loss(self, d):
#         alpha = 2.5
#         C = (self.lambda_c / (4 * np.pi)) ** 2
#         return C * d ** (-alpha)

#     def _generate_channels(self):
#         m = 3
#         d_Dn_R = np.linalg.norm(self.pos_Dn - self.pos_RIS)
#         d_Df_R = np.linalg.norm(self.pos_Df - self.pos_RIS)
#         d_R_BS = np.linalg.norm(self.pos_RIS - self.pos_BS)
#         d_T_R = np.linalg.norm(self.pos_T - self.pos_RIS)

#         self.h_DnR = self._nakagami(m, self._path_loss(d_Dn_R), self.L)
#         self.h_DfR = self._nakagami(m, self._path_loss(d_Df_R), self.L)
#         self.h_RB = self._nakagami(m, self._path_loss(d_R_BS), self.L)
#         self.h_TR = self._nakagami(m, self._path_loss(d_T_R), self.L)

#     def _sample_channels(self):
#         self._generate_channels()

#     def _transfer_partition(self, from_attr: str, to_attr: str):
#         donor = getattr(self, from_attr)
#         recip = getattr(self, to_attr)
#         new_donor = max(self.min_partition, donor - self.delta)
#         actual = donor - new_donor
#         setattr(self, from_attr, new_donor)
#         setattr(self, to_attr, recip + actual)

#     def _compute_metrics(self):
#         Ln = max(1, int(np.floor(self.a_n * self.L)))
#         Lf = max(1, int(np.floor(self.a_f * self.L)))
#         Lt = max(1, int(np.floor(self.a_t * self.L)))

#         # Transmit powers (Watts) as in system_model.m
#         Pn = 10 ** ((10 - 30) / 10)  # 10 dBm example (referenced against 30 dBm)
#         Pf = 10 ** (-self.delta_n / 10) * Pn
#         Pt = 10 ** ((20 - 30) / 10)  # 20 dBm

#         h_n_sum = np.sum(self.h_DnR[:Ln] * self.h_RB[:Ln])
#         h_f_sum = np.sum(self.h_DfR[Ln:Ln + Lf] * self.h_RB[Ln:Ln + Lf])
#         g_t_sum = np.sum(self.h_TR[-Lt:] * self.h_RB[-Lt:])
#         # MATLAB:
#         #   gT_eff = (sum(...))^2
#         #   I_radar = Pt * rho * abs(gT_eff)^2
#         gT_eff = g_t_sum ** 2

#         rho = (self.RCS * self.lambda_c ** 2 / (4 * np.pi) ** 3) * \
#               (2 * np.pi ** 2 / 12) * self.BW ** 2 * self.var_tau
        
#         I_radar = Pt * rho * (np.abs(gT_eff) ** 2)

#         # SINR Calculations
#         gamma_n = (np.abs(h_n_sum) ** 2 * Pn) / (
#             np.abs(h_f_sum) ** 2 * Pf + I_radar + self.sigma2
#         )
#         gamma_f = (np.abs(h_f_sum) ** 2 * Pf) / (
#             self.delta_n * np.abs(h_n_sum) ** 2 * Pn + I_radar + self.sigma2
#         )
#         gamma_bs = (rho * np.abs(gT_eff) ** 2 * Pt) / (
#             self.delta_n * np.abs(h_n_sum) ** 2 * Pn
#             + self.delta_f * np.abs(h_f_sum) ** 2 * Pf
#             + self.sigma2
#         )

#         rn = float(np.log2(1 + gamma_n))
#         rf = float(np.log2(1 + gamma_f))
#         asir = float((self.w / (2 * self.T_dur)) * np.log2(1 + 2 * self.T_dur * self.BW * gamma_bs))
#         jfi = jains_fairness_index([rn, rf])
#         reward = float(self.lambda_1 * jfi + self.lambda_2 * asir)

#         return rn, rf, asir, jfi, reward

#     def _get_state(self):
#         return np.array([
#             np.mean(np.abs(self.h_DnR)),
#             np.mean(np.abs(self.h_DfR)),
#             np.mean(np.abs(self.h_TR)),
#             self.a_n, self.a_f, self.a_t
#         ], dtype=np.float32)

#     def _apply_action(self, action: int):
#         if action == 0: self._transfer_partition("a_n", "a_f")
#         elif action == 1: self._transfer_partition("a_n", "a_t")
#         elif action == 2: self._transfer_partition("a_f", "a_n")
#         elif action == 3: self._transfer_partition("a_f", "a_t")
#         elif action == 4: self._transfer_partition("a_t", "a_n")
#         elif action == 5: self._transfer_partition("a_t", "a_f")

#         parts = np.clip([self.a_n, self.a_f, self.a_t], self.min_partition, 1.0)
#         parts = parts / np.sum(parts)
#         self.a_n, self.a_f, self.a_t = map(float, parts)

#     def step(self, action: int):
#         self.t += 1
#         self._apply_action(int(action))
#         self._sample_channels()

#         rn, rf, asir, jfi, reward = self._compute_metrics()
        
#         info = {
#             "r_n": rn, "r_f": rf, "asir": asir, "jfi": jfi,
#             "a_n": self.a_n, "a_f": self.a_f, "a_t": self.a_t
#         }
#         return self._get_state(), reward, (self.t >= self.max_steps), info

#     def step_continuous(self, action):
#         self.t += 1
#         parts = np.clip(np.atleast_1d(action).flatten(), self.min_partition, 1.0)
#         parts = parts / np.sum(parts)
#         self.a_n, self.a_f, self.a_t = map(float, parts)

#         self._sample_channels()
#         rn, rf, asir, jfi, reward = self._compute_metrics()
        
#         info = {
#             "r_n": rn, "r_f": rf, "asir": asir, "jfi": jfi,
#             "a_n": self.a_n, "a_f": self.a_f, "a_t": self.a_t
#         }
#         return self._get_state(), reward, (self.t >= self.max_steps), info

