%% RIS-Assisted ISAC-NOMA Uplink: Corrected Throughput & ASIR Computation
clear; clc;

%% 1. SYSTEM PARAMETERS
params.fc = 5.8e9;               % Carrier frequency (Hz)
params.c = 3e8;                  % Speed of light
params.lambda = params.c / params.fc;
params.BW = 10e6;                % Bandwidth (B)
params.sigma_BS_dBm = -104;      % Noise floor
params.sigma_BS_2 = 10^((params.sigma_BS_dBm-30)/10); 

% ISAC Specifics
params.RCS = 50;                 % Radar Cross Section
params.var_tau = 1e-6;           % Time delay fluctuation
params.w = 0.1;                  % Radar duty cycle
params.T = 1e-3;                 % Radar pulse duration

% NOMA ImpSIC and Power 
params.delta_n = 0.05;           % impSIC factor for Near User
params.delta_f = 0.05;           % impSIC factor for Far User
params.P_delta = 3.0;            % Power back-off for Far User (dB)
params.Pn_dBm = 10;              % Near User Transmit Power
params.Pt_dBm = 20;              % BS Sensing Power

% Positions (m)
params.pos_BS  = [0, 0];
params.pos_RIS = [100, 50];      
params.pos_Dn  = [598, 0];       % Near User 
params.pos_Df  = [748, 0];       % Far User 
params.pos_T   = [148, 100];     % Target [cite: 304]

% RIS Partitioning
params.L = 400;                  % Total REs
params.an = 0.33; params.af = 0.33; params.at = 0.34; % Equal split example

%% 2. MONTE CARLO SIMULATION
MC = 1000; % Number of realizations (reduce if slow, paper uses ~1e4)
R_n_total = 0; R_f_total = 0; ASIR_total = 0;

for k = 1:MC
    % Generate Channels (Nakagami-m)
    ch = generate_nakagami_channels(params);
    
    % Compute SINRs 
    [snr_n, snr_f, snr_bs] = compute_isac_sinr(ch, params);
    
    % Calculate Rates 
    R_n_total = R_n_total + log2(1 + snr_n); 
    R_f_total = R_f_total + log2(1 + snr_f); 
    
    % ASIR Calculation 
    BT_product = 2 * params.T * params.BW;
    term_asir = (params.w / (2 * params.T)) * log2(1 + BT_product * snr_bs); 
    ASIR_total = ASIR_total + term_asir;
end

% Final Averages
Ergodic_Rn = R_n_total / MC;
Ergodic_Rf = R_f_total / MC;
Ergodic_ASIR = ASIR_total / MC;

fprintf('--- Corrected Simulation Results ---\n');
fprintf('Near User Rate (Rn): %.4f bits/s/Hz\n', Ergodic_Rn);
fprintf('Far User Rate (Rf): %.4f bits/s/Hz\n', Ergodic_Rf);
fprintf('Average Sensing Info Rate (ASIR): %.4f bits/s/Hz\n', Ergodic_ASIR);

%% 3. CORE FUNCTIONS

function ch = generate_nakagami_channels(params)
    m = 3; % Nakagami-m shape parameter
    
    % Distances
    d_Dn_R = norm(params.pos_Dn - params.pos_RIS);
    d_Df_R = norm(params.pos_Df - params.pos_RIS);
    d_R_BS = norm(params.pos_RIS - params.pos_BS);
    d_T_R  = norm(params.pos_T - params.pos_RIS);
    
    % Path Loss: Omega = C * d^-alpha
    alpha = 2.5; 
    C = (params.lambda / (4*pi))^2;
    Omega = @(d) C * d^(-alpha);
    
    % Generate Small-scale fading (magnitude only, assuming optimal phase)
    gen_nak = @(L, d) sqrt(gamrnd(m, Omega(d)/m, [L, 1]));
    
    ch.h_DnR = gen_nak(params.L, d_Dn_R);
    ch.h_DfR = gen_nak(params.L, d_Df_R);
    ch.h_RB  = gen_nak(params.L, d_R_BS);
    ch.h_TR  = gen_nak(params.L, d_T_R);
end

function [gamma_n, gamma_f, gamma_bs] = compute_isac_sinr(ch, p)
    % Element Allocation
    Ln = floor(p.an * p.L); Lf = floor(p.af * p.L); Lt = floor(p.at * p.L);
    
    % Transmit Powers
    Pn = 10^((p.Pn_dBm - 30)/10); 
    Pf = 10^(-p.P_delta/10) * Pn; % Corrected NOMA power back-off
    Pt = 10^((p.Pt_dBm - 30)/10); 
    
    % Effective Channel Gains 
    h_Dn_eff = sum(ch.h_DnR(1:Ln) .* ch.h_RB(1:Ln));
    h_Df_eff = sum(ch.h_DfR(Ln+1:Ln+Lf) .* ch.h_RB(Ln+1:Ln+Lf));
    
    % Effective Target Channel
    gT_eff = (sum(ch.h_TR(end-Lt+1:end) .* ch.h_RB(end-Lt+1:end)))^2; 
    
    % Radar Echo scaling (rho)
    Cr = (p.RCS * p.lambda^2) / ((4 * pi)^3); [cite: 81]
    gamma_sq = (2 * pi^2) / 12; [cite: 81]
    rho = Cr * gamma_sq * (p.BW^2) * p.var_tau; [cite: 81]
    
    % Radar Calibration (undoing C^4 redundancy from cascading path loss)
    C_redundant = (p.lambda / (4 * pi))^2;
    radar_calibration = 1 / (C_redundant^4);
    
    % Received Radar Signal Power / Interference
    I_radar = Pt * rho * abs(gT_eff)^2 * radar_calibration;
    
    % SINR D_n (Near User)
    gamma_n = (abs(h_Dn_eff)^2 * Pn) / (abs(h_Df_eff)^2 * Pf + I_radar + p.sigma_BS_2); [cite: 88, 89]
    
    % SINR D_f (Far User - imperfect SIC)
    gamma_f = (abs(h_Df_eff)^2 * Pf) / (p.delta_n * abs(h_Dn_eff)^2 * Pn + I_radar + p.sigma_BS_2); [cite: 95, 96]
    
    % SINR BS (Radar Echo)
    gamma_bs = I_radar / (p.delta_n * abs(h_Dn_eff)^2 * Pn + p.delta_f * abs(h_Df_eff)^2 * Pf + p.sigma_BS_2); [cite: 99, 100]
end
