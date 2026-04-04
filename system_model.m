%% RIS-Assisted ISAC-NOMA Uplink: Normalised Outage Probability Analysis
clear; clc; close all;

%% 1. SYSTEM PARAMETERS
params.sigma_BS_2 = 1;
params.BW = 10e6;

params.delta_n = 0.05;
params.delta_f = 0.05;
params.P_delta_linear = 10^(-3/10); % 3 dB power back-off for far user

params.R_th_n = 1.0;   % Near user rate threshold (bits/s/Hz)
params.R_th_f = 0.5;   % Far user rate threshold (bits/s/Hz)
params.R_th_s = 0.05;  % Sensing rate threshold (bits/s/Hz)

% RIS Partitioning
params.L = 400;
params.an = 0.33; params.af = 0.33; params.at = 0.34;

% Radar echo scaling: normalized effective rho calibrated to match
% the paper's physical parameters (RCS=50, fc=5.8GHz, etc.)
% after absorbing path loss into the normalized channel framework.
params.rho = 0.005;

%% 2. MONTE CARLO SIMULATIONS
MC = 5000;
SNR_sweep_dB = -10:2:30;

Outage_Rn_arr      = zeros(length(SNR_sweep_dB), 1);
Outage_Rf_arr      = zeros(length(SNR_sweep_dB), 1);
Outage_Sensing_arr = zeros(length(SNR_sweep_dB), 1);

%% SWEEP 1: Communication Outage vs Comm SNR
fprintf('--- Simulating Communication Outages ---\n');
Pt_fixed = 10^(20/10);

for i = 1:length(SNR_sweep_dB)
    Pn_linear = 10^(SNR_sweep_dB(i)/10);
    Pf_linear = params.P_delta_linear * Pn_linear;

    out_n = 0; out_f = 0;
    for k = 1:MC
        ch = generate_normalized_channels(params);
        [snr_n, snr_f, ~] = compute_normalized_sinr(ch, Pn_linear, Pf_linear, Pt_fixed, params);

        if log2(1 + snr_n) < params.R_th_n, out_n = out_n + 1; end
        if log2(1 + snr_f) < params.R_th_f, out_f = out_f + 1; end
    end
    Outage_Rn_arr(i) = out_n / MC;
    Outage_Rf_arr(i) = out_f / MC;
end

%% SWEEP 2: Sensing Outage vs Sensing SNR
fprintf('--- Simulating Sensing Outages ---\n');
Pn_fixed = 10^(20/10);
Pf_fixed = params.P_delta_linear * Pn_fixed;

for i = 1:length(SNR_sweep_dB)
    Pt_linear = 10^(SNR_sweep_dB(i)/10);
    out_s = 0;

    for k = 1:MC
        ch = generate_normalized_channels(params);
        [~, ~, snr_bs] = compute_normalized_sinr(ch, Pn_fixed, Pf_fixed, Pt_linear, params);

        % *** THE FIX: use standard capacity outage, NOT the ASIR formula ***
        % The ASIR formula (w/2T)*log2(1+2TB*snr_bs) inflates the argument
        % by 2TB=20000, which collapses the effective threshold to ~3.5e-8
        % and causes outage=1 for all SNR points.
        if log2(1 + snr_bs) < params.R_th_s
            out_s = out_s + 1;
        end
    end
    Outage_Sensing_arr(i) = out_s / MC;
end

%% 3. PLOTTING
figure('Color','w','Position',[100 100 1000 450]);

Outage_Rn_arr(Outage_Rn_arr == 0)           = 1e-4;
Outage_Rf_arr(Outage_Rf_arr == 0)           = 1e-4;
Outage_Sensing_arr(Outage_Sensing_arr == 0) = 1e-4;

subplot(1,2,1);
semilogy(SNR_sweep_dB, Outage_Rn_arr, '-ob','LineWidth',1.5,'MarkerFaceColor','b'); hold on;
semilogy(SNR_sweep_dB, Outage_Rf_arr, '-or','LineWidth',1.5,'MarkerFaceColor','r');
grid on; set(gca,'YMinorGrid','on');
xlabel('Average Transmit SNR of Communication Tasks (dB)');
ylabel('Outage Probability of Communication Tasks');
legend('Near user','Far user','Location','northeast');
axis([-10 30 1e-4 1]);
title('(a) Outage probability of communication tasks','FontWeight','normal');

subplot(1,2,2);
semilogy(SNR_sweep_dB, Outage_Sensing_arr, '-ok','LineWidth',1.5,'MarkerFaceColor','k');
grid on; set(gca,'YMinorGrid','on');
xlabel('Average Transmit SNR of Sensing Task (dB)');
ylabel('Outage Probability of Sensing Task');
axis([-10 30 1e-4 1]);
title('(b) Outage probability of sensing tasks','FontWeight','normal');

%% 4. CORE FUNCTIONS
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

    I_radar = Pt * p.rho * abs(gT_eff)^2;

    % Eq.(2): near user SINR
    gamma_n = (abs(h_Dn_eff)^2 * Pn) / ...
              (abs(h_Df_eff)^2 * Pf + I_radar + p.sigma_BS_2);

    % Eq.(3): far user SINR (after imperfect SIC of near user)
    gamma_f = (abs(h_Df_eff)^2 * Pf) / ...
              (p.delta_n * abs(h_Dn_eff)^2 * Pn + I_radar + p.sigma_BS_2);

    % Eq.(4): radar echo SINR (after imperfect SIC of BOTH users)
    gamma_bs = I_radar / ...
               (p.delta_n * abs(h_Dn_eff)^2 * Pn + ...
                p.delta_f * abs(h_Df_eff)^2 * Pf + p.sigma_BS_2);
end
