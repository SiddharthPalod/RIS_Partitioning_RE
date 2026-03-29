%% RIS-Assisted ISAC-NOMA Uplink: Throughput & ASIR Computation
% Based on performance analysis under imperfect SIC and Nakagami-m fading.

clear; clc;

%% 1. SYSTEM PARAMETERS (Ref: Paper 4 Layout)
params.fc = 5.8e9;               % Carrier frequency (Hz) [cite: 226]
params.c = 3e8;                  % Speed of light
params.lambda = params.c / params.fc;
params.BW = 10e6;                % Bandwidth (B) [cite: 226]
params.sigma_BS_dBm = -104;      % Noise floor [cite: 226]
params.sigma_BS_2 = 10^((params.sigma_BS_dBm-30)/10); 

% ISAC Specifics
params.RCS = 50;                 % Radar Cross Section [cite: 226]
params.var_tau = 1e-6;           % Time delay fluctuation [cite: 226]
params.w = 0.1;                  % Radar duty cycle (varpi) 
params.T = 1e-3;                 % Radar pulse duration 
params.delta_n = 0.05;           % impSIC factor for Near User [cite: 220]
params.delta_f = 0.05;           % impSIC factor for Far User [cite: 100]

% Positions (m) [cite: 377-388]
params.pos_BS  = [0, 0];
params.pos_RIS = [100, 50];      % [cite: 226]
params.pos_Dn  = [598, 0];       % Near User [cite: 303]
params.pos_Df  = [748, 0];       % Far User [cite: 303]
params.pos_T   = [148, 100];     % Target [cite: 304]

% RIS Partitioning [cite: 64]
params.L = 400;                  % Total REs
params.an = 0.1; params.af = 0.4; params.at = 0.5; % Example coefficients [cite: 262]

%% 2. MONTE CARLO SIMULATION
MC = 1e4; % Number of realizations
R_n_total = 0; R_f_total = 0; ASIR_total = 0;

for k = 1:MC
    % Generate Channels (Nakagami-m)
    ch = generate_nakagami_channels(params);
    
    % Compute SINRs (Equations 2, 3, 4)
    [snr_n, snr_f, snr_bs] = compute_isac_sinr(ch, params);
    
    % Calculate Rates (Ref: Paper 4 wrapper style)
    R_n_total = R_n_total + log2(1 + snr_n); % [cite: 251]
    R_f_total = R_f_total + log2(1 + snr_f); % [cite: 255]
    
    % ASIR Calculation (Ref: Paper 1, Eq 18)
    % R_avg = (w/2T) * log2(1 + 2TB * gamma_BS)
    term_asir = (params.w / (2 * params.T)) * log2(1 + 2 * params.T * params.BW * snr_bs); % 
    ASIR_total = ASIR_total + term_asir;
end

% Final Averages
Ergodic_Rn = R_n_total / MC;
Ergodic_Rf = R_f_total / MC;
Ergodic_ASIR = ASIR_total / MC;

fprintf('--- Simulation Results ---\n');
fprintf('Near User Rate (Rn): %.4f bits/s/Hz\n', Ergodic_Rn);
fprintf('Far User Rate (Rf): %.4f bits/s/Hz\n', Ergodic_Rf);
fprintf('Average Sensing Info Rate (ASIR): %.4f bits/s/Hz\n', Ergodic_ASIR);

%% 3. CORE FUNCTIONS

function ch = generate_nakagami_channels(params)
    % Nakagami-m shape parameter
    m = 3; % [cite: 226]
    
    % Calculate Distances
    d_Dn_R = norm(params.pos_Dn - params.pos_RIS);
    d_Df_R = norm(params.pos_Df - params.pos_RIS);
    d_R_BS = norm(params.pos_RIS - params.pos_BS);
    d_T_R  = norm(params.pos_T - params.pos_RIS);
    
    % Path Loss: Omega = C * d^-alpha [cite: 62]
    alpha = 2.5; C = (params.lambda / (4*pi))^2;
    Omega = @(d) C * d^(-alpha);
    
    % Generate Small-scale fading (Nakagami-m distributed)
    % Helper: Nakagami is sqrt of Gamma
    gen_nak = @(L, d) sqrt(gamrnd(m, Omega(d)/m, [L, 1]));
    
    ch.h_DnR = gen_nak(params.L, d_Dn_R);
    ch.h_DfR = gen_nak(params.L, d_Df_R);
    ch.h_RB  = gen_nak(params.L, d_R_BS);
    ch.h_TR  = gen_nak(params.L, d_T_R);
end

function [gamma_n, gamma_f, gamma_bs] = compute_isac_sinr(ch, p)
    % Element Allocation [cite: 64]
    Ln = floor(p.an * p.L); Lf = floor(p.af * p.L); Lt = floor(p.at * p.L);
    
    % Transmit Powers [cite: 75]
    Pn = 10^((10-30)/10); % 10 dBm example
    Pf = 10^(-p.delta_n/10) * Pn; % Power back-off
    Pt = 10^((20-30)/10); % BS Sensing Power 20 dBm
    
    % Effective Channel Gains (Assuming Optimal Phase Shifts) [cite: 92, 97, 100]
    % For NOMA, users align with their respective RIS zones
    h_Dn_eff = sum(ch.h_DnR(1:Ln) .* ch.h_RB(1:Ln));
    h_Df_eff = sum(ch.h_DfR(Ln+1:Ln+Lf) .* ch.h_RB(Ln+1:Ln+Lf));
    
    % Radar Echo scaling (rho) [cite: 81]
    % Simplified rho based on RCS and BW
    rho = (p.RCS * p.lambda^2 / (4*pi)^3) * (2*pi^2/12) * p.BW^2 * p.var_tau;
    gT_eff = sum(ch.h_TR(end-Lt+1:end) .* ch.h_RB(end-Lt+1:end))^2; 
    
    % Interference Terms
    I_radar = Pt * rho * abs(gT_eff)^2;
    
    % SINR D_n (Near User) [cite: 88]
    gamma_n = (abs(h_Dn_eff)^2 * Pn) / (abs(h_Df_eff)^2 * Pf + I_radar + p.sigma_BS_2);
    
    % SINR D_f (Far User - imperfect SIC) [cite: 95]
    gamma_f = (abs(h_Df_eff)^2 * Pf) / (p.delta_n * abs(h_Dn_eff)^2 * Pn + I_radar + p.sigma_BS_2);
    
    % SINR BS (Radar Echo - after comm decoding) [cite: 100]
    gamma_bs = (rho * abs(gT_eff)^2 * Pt) / (p.delta_n * abs(h_Dn_eff)^2 * Pn + p.delta_f * abs(h_Df_eff)^2 * Pf + p.sigma_BS_2);
end
