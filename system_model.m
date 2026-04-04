%% RIS-Assisted ISAC-NOMA Uplink: Full Simulation (Rates + Outage)
clear; clc; close all;

%% =========================================================
%% SECTION A: PHYSICAL PARAMETERS (for Rate/ASIR sweep)
%% =========================================================
params.fc = 5.8e9;
params.c = 3e8;
params.lambda = params.c / params.fc;
params.BW = 10e6;
params.sigma_BS_dBm = -104;
params.sigma_BS_2 = 10^((params.sigma_BS_dBm-30)/10); 

params.RCS = 50;
params.var_tau = 1e-6;
params.w = 0.1;
params.T = 1e-3;

params.delta_n = 0.05;
params.delta_f = 0.05;
params.P_delta = 3.0;
params.Pt_dBm = 20;

params.pos_BS  = [0, 0];
params.pos_RIS = [100, 50];      
params.pos_Dn  = [598, 0];
params.pos_Df  = [748, 0];
params.pos_T   = [148, 100];

params.L = 400;
params.an = 0.33; params.af = 0.33; params.at = 0.34;

%% =========================================================
%% SECTION B: NORMALIZED PARAMETERS (for Outage sweep)
%% =========================================================
norm_params.sigma_BS_2     = 1;
norm_params.BW             = 10e6;
norm_params.delta_n        = 0.05;
norm_params.delta_f        = 0.05;
norm_params.P_delta_linear = 10^(-3/10);
norm_params.R_th_n         = 1.0;
norm_params.R_th_f         = 0.5;
norm_params.R_th_s         = 0.05;
norm_params.L              = 400;
norm_params.an             = 0.33; norm_params.af = 0.33; norm_params.at = 0.34;
norm_params.rho            = 0.005;

%% =========================================================
%% SWEEP 1: Ergodic Communication Rate vs Pn
%% =========================================================
MC = 1000;
Pn_sweep_dBm = 0:5:30;

Ergodic_Rn_arr = zeros(length(Pn_sweep_dBm), 1);
Ergodic_Rf_arr = zeros(length(Pn_sweep_dBm), 1);

fprintf('--- Communication Rate Sweep (vs Pn) ---\n');
for i = 1:length(Pn_sweep_dBm)
    params.Pn_dBm = Pn_sweep_dBm(i);
    R_n_total = 0; R_f_total = 0;
    for k = 1:MC
        ch = generate_nakagami_channels(params);
        [snr_n, snr_f, ~] = compute_isac_sinr(ch, params);
        R_n_total = R_n_total + log2(1 + snr_n); 
        R_f_total = R_f_total + log2(1 + snr_f); 
    end
    Ergodic_Rn_arr(i) = R_n_total / MC;
    Ergodic_Rf_arr(i) = R_f_total / MC;
    fprintf('Pn = %2d dBm | Rn: %6.4f | Rf: %6.4f\n', ...
        params.Pn_dBm, Ergodic_Rn_arr(i), Ergodic_Rf_arr(i));
end

%% =========================================================
%% SWEEP 2: ASIR vs Pt (Pn fixed)
%% =========================================================
Pt_sweep_dBm = 10:5:40;
params.Pn_dBm = 10;

Ergodic_ASIR_arr = zeros(length(Pt_sweep_dBm), 1);

fprintf('--- ASIR Sweep (vs Pt, Pn fixed at %d dBm) ---\n', params.Pn_dBm);
for i = 1:length(Pt_sweep_dBm)
    params.Pt_dBm = Pt_sweep_dBm(i);
    ASIR_total = 0;
    for k = 1:MC
        ch = generate_nakagami_channels(params);
        [~, ~, snr_bs] = compute_isac_sinr(ch, params);
        BT_product = 2 * params.T * params.BW;
        ASIR_total = ASIR_total + (params.w / (2*params.T)) * log2(1 + BT_product * snr_bs);
    end
    Ergodic_ASIR_arr(i) = ASIR_total / MC;
    fprintf('Pt = %2d dBm | ASIR: %8.4f\n', params.Pt_dBm, Ergodic_ASIR_arr(i));
end

%% =========================================================
%% SWEEP 3: Communication Outage vs Comm SNR (normalized)
%% =========================================================
MC_out = 5000;
SNR_sweep_dB = -10:2:30;

Outage_Rn_arr      = zeros(length(SNR_sweep_dB), 1);
Outage_Rf_arr      = zeros(length(SNR_sweep_dB), 1);
Outage_Sensing_arr = zeros(length(SNR_sweep_dB), 1);

fprintf('--- Simulating Communication Outages ---\n');
Pt_fixed = 10^(20/10);

for i = 1:length(SNR_sweep_dB)
    Pn_linear = 10^(SNR_sweep_dB(i)/10);
    Pf_linear = norm_params.P_delta_linear * Pn_linear;

    out_n = 0; out_f = 0;
    for k = 1:MC_out
        ch = generate_normalized_channels(norm_params);
        [snr_n, snr_f, ~] = compute_normalized_sinr(ch, Pn_linear, Pf_linear, Pt_fixed, norm_params);
        if log2(1 + snr_n) < norm_params.R_th_n, out_n = out_n + 1; end
        if log2(1 + snr_f) < norm_params.R_th_f, out_f = out_f + 1; end
    end
    Outage_Rn_arr(i) = out_n / MC_out;
    Outage_Rf_arr(i) = out_f / MC_out;
end

%% =========================================================
%% SWEEP 4: Sensing Outage vs Sensing SNR (normalized)
%% =========================================================
fprintf('--- Simulating Sensing Outages ---\n');
Pn_fixed = 10^(20/10);
Pf_fixed = norm_params.P_delta_linear * Pn_fixed;

for i = 1:length(SNR_sweep_dB)
    Pt_linear = 10^(SNR_sweep_dB(i)/10);
    out_s = 0;
    for k = 1:MC_out
        ch = generate_normalized_channels(norm_params);
        [~, ~, snr_bs] = compute_normalized_sinr(ch, Pn_fixed, Pf_fixed, Pt_linear, norm_params);
        if log2(1 + snr_bs) < norm_params.R_th_s
            out_s = out_s + 1;
        end
    end
    Outage_Sensing_arr(i) = out_s / MC_out;
end

%% =========================================================
%% PLOTTING
%% =========================================================

%% Figure 1: Communication Rates and ASIR
figure('Name', 'ISAC-NOMA Uplink Performance', 'Color', 'w');

subplot(1, 2, 1);
plot(Pn_sweep_dBm, Ergodic_Rn_arr, '-o', 'LineWidth', 2, 'MarkerSize', 6); hold on;
plot(Pn_sweep_dBm, Ergodic_Rf_arr, '-s', 'LineWidth', 2, 'MarkerSize', 6);
grid on;
xlabel('Near User Transmit Power, P_n (dBm)', 'FontWeight', 'bold');
ylabel('Achievable Rate (bits/s/Hz)', 'FontWeight', 'bold');
title('Communication Rates vs. Transmit Power');
legend('Near User (R_n)', 'Far User (R_f)', 'Location', 'best');

subplot(1, 2, 2);
plot(Pt_sweep_dBm, Ergodic_ASIR_arr, '-^', 'LineWidth', 2, 'MarkerSize', 6, 'Color', '#D95319');
grid on;
xlabel('BS Sensing Transmit Power, P_t (dBm)', 'FontWeight', 'bold');
ylabel('ASIR (bits/s)', 'FontWeight', 'bold');
title(sprintf('Average Sensing Info Rate vs. P_t  (P_n = %d dBm)', params.Pn_dBm));
legend('Target Sensing (ASIR)', 'Location', 'best');

%% Figure 2: Outage Probabilities
Outage_Rn_arr(Outage_Rn_arr == 0)           = 1e-4;
Outage_Rf_arr(Outage_Rf_arr == 0)           = 1e-4;
Outage_Sensing_arr(Outage_Sensing_arr == 0) = 1e-4;

figure('Color', 'w', 'Position', [100, 100, 1000, 450]);

subplot(1, 2, 1);
semilogy(SNR_sweep_dB, Outage_Rn_arr, '-ob', 'LineWidth', 1.5, 'MarkerFaceColor', 'b'); hold on;
semilogy(SNR_sweep_dB, Outage_Rf_arr, '-or', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
grid on; set(gca, 'YMinorGrid', 'on');
xlabel('Average Transmit SNR of Communication Tasks (dB)');
ylabel('Outage Probability of Communication Tasks');
legend('Near user', 'Far user', 'Location', 'northeast');
axis([-10 30 1e-4 1]);
title('(a) Outage probability of communication tasks', 'FontWeight', 'normal');

subplot(1, 2, 2);
semilogy(SNR_sweep_dB, Outage_Sensing_arr, '-ok', 'LineWidth', 1.5, 'MarkerFaceColor', 'k');
grid on; set(gca, 'YMinorGrid', 'on');
xlabel('Average Transmit SNR of Sensing Task (dB)');
ylabel('Outage Probability of Sensing Task');
axis([-10 30 1e-4 1]);
title('(b) Outage probability of sensing tasks', 'FontWeight', 'normal');

%% =========================================================
%% FUNCTIONS
%% =========================================================

function ch = generate_nakagami_channels(params)
    m = 3;
    d_Dn_R = norm(params.pos_Dn - params.pos_RIS);
    d_Df_R = norm(params.pos_Df - params.pos_RIS);
    d_R_BS = norm(params.pos_RIS - params.pos_BS);
    d_T_R  = norm(params.pos_T  - params.pos_RIS);
    alpha = 2.5; 
    C = (params.lambda / (4*pi))^2;
    Omega = @(d) C * d^(-alpha);
    gen_nak = @(L, d) sqrt(gamrnd(m, Omega(d)/m, [L, 1]));
    ch.h_DnR = gen_nak(params.L, d_Dn_R);
    ch.h_DfR = gen_nak(params.L, d_Df_R);
    ch.h_RB  = gen_nak(params.L, d_R_BS);
    ch.h_TR  = gen_nak(params.L, d_T_R);
end

function [gamma_n, gamma_f, gamma_bs] = compute_isac_sinr(ch, p)
    Ln = floor(p.an * p.L); Lf = floor(p.af * p.L); Lt = floor(p.at * p.L);
    Pn = 10^((p.Pn_dBm - 30)/10); 
    Pf = 10^(-p.P_delta/10) * Pn;
    Pt = 10^((p.Pt_dBm - 30)/10); 
    h_Dn_eff = sum(ch.h_DnR(1:Ln) .* ch.h_RB(1:Ln));
    h_Df_eff = sum(ch.h_DfR(Ln+1:Ln+Lf) .* ch.h_RB(Ln+1:Ln+Lf));
    gT_eff   = (sum(ch.h_TR(end-Lt+1:end) .* ch.h_RB(end-Lt+1:end)))^2; 
    Cr = (p.RCS * p.lambda^2) / ((4*pi)^3);
    gamma_sq = (2*pi^2) / 12;
    rho = Cr * gamma_sq * (p.BW^2) * p.var_tau;
    C_redundant = (p.lambda / (4*pi))^2;
    radar_calibration = 1 / (C_redundant^4);
    I_radar = Pt * rho * abs(gT_eff)^2 * radar_calibration;
    gamma_n  = (abs(h_Dn_eff)^2 * Pn) / (abs(h_Df_eff)^2 * Pf + I_radar + p.sigma_BS_2);
    gamma_f  = (abs(h_Df_eff)^2 * Pf) / (p.delta_n * abs(h_Dn_eff)^2 * Pn + I_radar + p.sigma_BS_2);
    gamma_bs = I_radar / (p.delta_n * abs(h_Dn_eff)^2 * Pn + p.delta_f * abs(h_Df_eff)^2 * Pf + p.sigma_BS_2);
end

function ch = generate_normalized_channels(params)
    m = 3;
    gen_nak = @(L) sqrt(gamrnd(m, 1/m, [L, 1]));
    ch.h_DnR = gen_nak(params.L);
    ch.h_DfR = gen_nak(params.L);
    ch.h_RB  = gen_nak(params.L);
    ch.h_TR  = gen_nak(params.L);
end

function [gamma_n, gamma_f, gamma_bs] = compute_normalized_sinr(ch, Pn, Pf, Pt, p)
    Ln = floor(p.an * p.L);
    Lf = floor(p.af * p.L);
    Lt = p.L - Ln - Lf;
    h_Dn_eff = sum(ch.h_DnR(1:Ln) .* ch.h_RB(1:Ln)) / sqrt(Ln);
    h_Df_eff = sum(ch.h_DfR(Ln+1:Ln+Lf) .* ch.h_RB(Ln+1:Ln+Lf)) / sqrt(Lf);
    gT_eff   = sum(ch.h_TR(end-Lt+1:end) .* ch.h_RB(end-Lt+1:end)) / sqrt(Lt);
    I_radar  = Pt * p.rho * abs(gT_eff)^2;
    gamma_n  = (abs(h_Dn_eff)^2 * Pn) / (abs(h_Df_eff)^2 * Pf + I_radar + p.sigma_BS_2);
    gamma_f  = (abs(h_Df_eff)^2 * Pf) / (p.delta_n * abs(h_Dn_eff)^2 * Pn + I_radar + p.sigma_BS_2);
    gamma_bs = I_radar / (p.delta_n * abs(h_Dn_eff)^2 * Pn + p.delta_f * abs(h_Df_eff)^2 * Pf + p.sigma_BS_2);
end
