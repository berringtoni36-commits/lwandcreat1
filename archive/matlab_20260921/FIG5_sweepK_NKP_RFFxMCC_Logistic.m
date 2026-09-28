% ============================================================
% FIG5_K_Sweep_MCCk_FIXED.m
% Fig.5  Effect of MCC parameter k (kappa) under Logistic chaotic noise
% Method: RFF + FxNLMS + MCC
%
% --------- Key fixes vs your version ----------
% (1) kappa is NOT used as a linear gain (which acts like scaling step-size).
%     Use kappa inside the exponent:
%         ge = exp( -kappa * e^2 / (2*sigma^2) )
%     so kappa changes "robustness strength", not "update amplitude".
%
% (2) Use feature-domain filtered-x for correct FXLMS gradient:
%         phif = sum_i Sw(i) * phi(n-i+1)
%     (equivalent to filtering each feature dimension by S(z)).
%
% (3) Monte Carlo is made meaningful by randomizing logistic initial condition u0.
% ============================================================

clear; close all; clc;

%% ---------------- Global settings ----------------
T       = 20000;        % iterations (0~2x10^4)
inde    = 20;           % Monte Carlo trials
M       = 20;           % tapped-delay length
fg      = 0.999;        % ANR smoothing factor
eps_n   = 1e-10;
eps_anr = 1e-12;

rng(871,'twister');     % reproducible

%% ---------------- Paths ----------------
Pw = [0 0 0 1 -0.3 0.2];      % primary path
Sw = [0 0 1 0.5];             % secondary path (minimum-phase)
Ls = numel(Sw);

%% ---------------- RFF / MCC params ----------------
D      = 500;        % RFF feature dimension
sigma  = 2.0;        % MCC kernel width (same as your Fig6)
mu_mcc = 0.8;        % step-size (建议先用 0.5~1.0；你原来 2*k 会太猛)

% kappa list
kappaList = [1.0 1.5 2.0 2.8 4.0 5.6 6.0 8.0 12 15 20];

%% ---------------- Random Fourier Features ----------------
gamma_rff = 1.0;
W = gamma_rff * randn(D, M);
b = 2*pi*rand(D, 1);
rff_map = @(xvec) sqrt(2/D) * cos(W*xvec + b);

%% ---------------- Storage ----------------
Knum = numel(kappaList);
ANR_all = zeros(T, Knum);

%% ================= Sweep over kappa =================
for ik = 1:Knum
    kappa = kappaList(ik);

    ANR_acc = zeros(T,1);

    for it = 1:inde
        % ==========================================================
        % (1) Logistic chaotic input (randomize u0 each trial)
        % ==========================================================
        u0 = 0.1 + 0.8*rand;                 % random initial in (0,1)
        u = logistic_delay6(T, u0);
        u = u - mean(u);
        u = u / (std(u) + 1e-12);

        % ==========================================================
        % (2) Nonlinear disturbance:
        % g = P(z)*u
        % d(n)=g(n-2)+0.08*g^2(n-2)-0.04*g^3(n-1)
        % ==========================================================
        g0 = filter(Pw, 1, u);

        d = zeros(T,1);
        n0 = 3:T;
        d(n0) = g0(n0-2) + 0.08*(g0(n0-2).^2) - 0.04*(g0(n0-1).^3);

        % ----- init -----
        w = zeros(D,1);

        xbuf  = zeros(M,1);          % tapped delay of u
        ybuf  = zeros(Ls,1);         % tapped delay of y (for S(z))

        % feature-domain buffer for filtered-x
        phibuf = zeros(D, Ls);       % [phi(n), phi(n-1), ..., phi(n-Ls+1)]

        Pd = 0; Pe = 0;
        ANR = zeros(T,1);

        for n = 1:T
            % tapped-delay vectors
            xbuf  = [u(n);  xbuf(1:end-1)];

            % RFF features (from xbuf)
            phi  = rff_map(xbuf);

            % controller output (before secondary path)
            y  = w' * phi;

            % secondary path on control signal (physical path)
            ybuf = [y; ybuf(1:end-1)];
            ys = Sw * ybuf;          % y_s(n)

            % error
            e = d(n) - ys;

            % -----------------------------
            % FIX 1: MCC weight uses kappa in exponent (not as linear gain)
            % -----------------------------
            ge = exp( -kappa * (e^2) / (2*sigma^2) );

            % -----------------------------
            % FIX 2: feature-domain filtered-x for correct Fx gradient
            % phif = sum_i Sw(i) * phi(n-i+1)
            % -----------------------------
            phibuf = [phi, phibuf(:,1:Ls-1)];
            phif = phibuf * Sw(:);   % D×1

            % NLMS+MCC update
            den = (phif'*phif + eps_n);
            w = w + mu_mcc * ge * (e/den) * phif;

            % ANR
            Pd = fg*Pd + (1-fg)*(d(n)^2);
            Pe = fg*Pe + (1-fg)*(e^2);
            ANR(n) = 10*log10( (Pe+eps_anr)/(Pd+eps_anr) );
        end

        ANR_acc = ANR_acc + ANR;
    end

    ANR_all(:,ik) = ANR_acc / inde;
end

%% ---------------- Plot (paper-like multi-curves) ----------------
t  = 0:T-1;
mk = round(linspace(1,T,11));

figure('Color',[0.94 0.94 0.94]); hold on; grid on; box on;
set(gca,'Color',[0.94 0.94 0.94],'FontName','Times New Roman',...
    'FontSize',18,'LineWidth',1.2);

% color map
try
    cols = turbo(Knum);
catch
    cols = lines(Knum);
end

for ik = 1:Knum
    plot(t, ANR_all(:,ik), ...
        'LineWidth', 2.2, ...
        'Color', cols(ik,:), ...
        'Marker','o', 'MarkerIndices', mk, ...
        'MarkerSize', 9, ...
        'MarkerFaceColor', cols(ik,:), ...
        'DisplayName', sprintf('k=%.1f', kappaList(ik)));
end

xlabel('Iteration','FontName','Times New Roman','FontSize',22);
ylabel('ANR (dB)','FontName','Times New Roman','FontSize',22);
title('Fig.5  Effect of MCC parameter k','FontName','Times New Roman','FontSize',24);

xlim([0 2e4]);
ylim([-35 5]);

ax = gca;
ax.XAxis.Exponent = 4;   % show ×10^4
legend('Location','northeast');

%% ---------------- Steady-state (last 5000) ----------------
Lss = 5000;
ss = mean(ANR_all(end-Lss+1:end,:), 1);

figure('Color',[0.94 0.94 0.94]); hold on; grid on; box on;
set(gca,'Color',[0.94 0.94 0.94],'FontName','Times New Roman',...
    'FontSize',18,'LineWidth',1.2);

plot(kappaList, ss, '-o', 'LineWidth',2.4, 'MarkerSize',9);
xlabel('k','FontName','Times New Roman','FontSize',22);
ylabel('Averaged steady-state ANR (dB)','FontName','Times New Roman','FontSize',22);
title('Steady-state ANR vs k','FontName','Times New Roman','FontSize',24);

%% ================= helper =================
function u = logistic_delay6(T, u0)
    % u(n)=4*u(n-6)*(1-u(n-6))
    u = zeros(T,1);
    u(1:6) = u0;
    for n = 7:T
        u(n) = 4*u(n-6)*(1-u(n-6));
    end
end