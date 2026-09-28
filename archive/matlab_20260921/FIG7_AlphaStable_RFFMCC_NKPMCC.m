% ============================================================
% FIG7_Logistic_RFFMCC_NKPMCC_PaperStyle.m
% Fig.7 Averaged steady-state ANR for Logistic chaotic noise source
% (paper-like bar style, two algorithms in different colors)
%
% Curves used to compute steady-state:
%   1) RFF+NLMS+MCC (baseline)
%   2) NKP+NLMS+MCC (proposed)
%
% IMPORTANT: This script re-runs Fig6 setting to get steady-state ANR.
% ============================================================

clear; close all; clc;

%% ---------------- Global settings (align with your Fig6) ----------------
T       = 20000;        % 0 ~ 2e4
inde    = 20;           % MC trials
M       = 20;           % tapped-delay length
fg      = 0.999;        % ANR smoothing
eps_n   = 1e-10;
eps_anr = 1e-12;

seed0 = 871;
rng(seed0,'twister');

% steady-state window (last part)
ss_len = 5000;                 % you can change: 3000/2000 to match paper
ss_idx = (T-ss_len+1):T;

%% ---------------- Paths (your setting) ----------------
Pw = [0 0 0 1 -0.3 0.2];       % primary path
Sw = [0 0 1 0.5];              % secondary path
Ls = numel(Sw);

%% ---------------- RFF / MCC / NKP params (your Fig6) ----------------
D      = 500;
sigma  = 2.0;
kappa  = 2;

mu_mcc = 2;                    % baseline step-size
mu_nkp = 2;                    % NKP step-size

% NKP block preconditioning (paper-aligned)
Kblk    = 4;
Bblk    = floor(D / Kblk);
alpha_p = 0.99;

%% ---------------- Random Fourier Features ----------------
gamma_rff = 1.0;
W = gamma_rff * randn(D, M);
b = 2*pi*rand(D, 1);
rff_map = @(xvec) sqrt(2/D) * cos(W*xvec + b);

%% ---------------- Steady-state accumulators ----------------
ss_mcc_acc = 0;
ss_nkp_acc = 0;

%% ================= Monte Carlo =================
for it = 1:inde
    rng(seed0 + it - 1, 'twister');

    % NKP running block power init
    pblk = ones(Kblk,1);

    % (1) Logistic chaotic input (paper): u(n)=4*u(n-6)*(1-u(n-6))
    u = logistic_delay6(T, 0.37);
    u = u - mean(u);
    u = u / (std(u) + 1e-12);

    % (2) Nonlinear disturbance (paper):
    % g = P(z)*u
    % d(n)=g(n-2)+0.08*g^2(n-2)-0.04*g^3(n-1)
    gsig = filter(Pw, 1, u);
    d = zeros(T,1);
    n0 = 3:T;
    d(n0) = gsig(n0-2) + 0.08*(gsig(n0-2).^2) - 0.04*(gsig(n0-1).^3);

    % filtered reference
    xf = filter(Sw, 1, u);

    % weights
    w_mcc = zeros(D,1);
    w_nkp = zeros(D,1);

    % buffers
    xbuf  = zeros(M,1);
    xfbuf = zeros(M,1);
    ybuf_mcc = zeros(Ls,1);
    ybuf_nkp = zeros(Ls,1);

    % ANR smoothing powers
    Pd_mcc = 0; Pe_mcc = 0;
    Pd_nkp = 0; Pe_nkp = 0;

    ANR_mcc = zeros(T,1);
    ANR_nkp = zeros(T,1);

    for n = 1:T
        xbuf  = [u(n);  xbuf(1:end-1)];
        xfbuf = [xf(n); xfbuf(1:end-1)];

        phi  = rff_map(xbuf);
        phif = rff_map(xfbuf);

        % -------- baseline: RFF + NLMS + MCC --------
        y_mcc = w_mcc' * phi;
        ybuf_mcc = [y_mcc; ybuf_mcc(1:end-1)];
        ys_mcc = Sw * ybuf_mcc;
        e_mcc  = d(n) - ys_mcc;

        gm   = kappa * exp( - (e_mcc^2) / (2*sigma^2) );
        den  = (phif'*phif + eps_n);
        w_mcc = w_mcc + mu_mcc * gm * (e_mcc/den) * phif;

        Pd_mcc = fg*Pd_mcc + (1-fg)*(d(n)^2);
        Pe_mcc = fg*Pe_mcc + (1-fg)*(e_mcc^2);
        ANR_mcc(n) = 10*log10( (Pd_mcc+eps_anr)/(Pe_mcc+eps_anr) );

        % -------- proposed: NKP + NLMS + MCC --------
        [psi, psif, pblk] = nkp_block_precond_running(phi, phif, Kblk, Bblk, pblk, alpha_p, eps_n);

        y_nkp = w_nkp' * psi;
        ybuf_nkp = [y_nkp; ybuf_nkp(1:end-1)];
        ys_nkp = Sw * ybuf_nkp;
        e_nkp  = d(n) - ys_nkp;

        gn   = kappa * exp( - (e_nkp^2) / (2*sigma^2) );
        den2 = (psif'*psif + eps_n);
        w_nkp = w_nkp + mu_nkp * gn * (e_nkp/den2) * psif;

        Pd_nkp = fg*Pd_nkp + (1-fg)*(d(n)^2);
        Pe_nkp = fg*Pe_nkp + (1-fg)*(e_nkp^2);
        ANR_nkp(n) = 10*log10( (Pd_nkp+eps_anr)/(Pe_nkp+eps_anr) );
    end

    ss_mcc_acc = ss_mcc_acc + mean(ANR_mcc(ss_idx), 'omitnan');
    ss_nkp_acc = ss_nkp_acc + mean(ANR_nkp(ss_idx), 'omitnan');

    fprintf('[MC %d/%d] done\n', it, inde);
end

ss_mcc_avg = ss_mcc_acc / inde;
ss_nkp_avg = ss_nkp_acc / inde;

%% ================= Plot Fig.7 (paper-like bar, two colors) =================
vals = [ss_mcc_avg, ss_nkp_avg];

% paper-like colors (close to your example figure palette)
col_baseline = [0.68 0.45 0.60];   % purple-ish
col_nkp      = [0.33 0.65 0.65];   % teal-ish

figure('Color',[0.90 0.90 0.90]); clf; hold on; box on; grid on;
ax = gca;
set(ax,'Color',[0.90 0.90 0.90], ...
    'FontName','Times New Roman','FontSize',18,'LineWidth',1.2);

bh = bar(1:2, vals, 0.75, 'FaceColor','flat', 'EdgeColor',[0 0 0], 'LineWidth',1.2);
bh.CData(1,:) = col_baseline;
bh.CData(2,:) = col_nkp;

set(ax,'XTick',1:2, 'XTickLabel',{'RFF+NLMS+MCC','NKP+NLMS+MCC'});

ylabel('Averaged steady-state ANR','FontName','Times New Roman','FontSize',22);
title('Fig. 7.  Averaged steady-state ANR for Logistic chaotic noise source.', ...
    'FontName','Times New Roman','FontSize',22);

ylim([-30 0]);               % match paper Fig7 look
xlim([0.2 2.8]);

% legend like paper (bottom-left, boxed)
h1 = patch(NaN,NaN,col_baseline,'EdgeColor',[0 0 0],'LineWidth',1.0);
h2 = patch(NaN,NaN,col_nkp,'EdgeColor',[0 0 0],'LineWidth',1.0);
lgd = legend([h1 h2], {'RFF+NLMS+MCC (baseline)','NKP+NLMS+MCC (proposed)'}, ...
    'Location','southwest');
set(lgd,'FontName','Times New Roman','FontSize',16,'LineWidth',1.2);

% caption style (optional)
% annotation('textbox',[0.05 0.01 0.9 0.06],'String', ...
%     'Fig. 7.  Averaged steady-state ANR of various algorithms for Logistic chaotic noise source.', ...
%     'EdgeColor','none','HorizontalAlignment','center', ...
%     'FontName','Times New Roman','FontSize',16,'FontWeight','bold');

%% ================= helper functions =================
function u = logistic_delay6(T, u0)
    % paper: u(n)=4*u(n-6)*(1-u(n-6))
    u = zeros(T,1);
    u(1:6) = u0;
    for n = 7:T
        u(n) = 4*u(n-6)*(1-u(n-6));
    end
end

function [psi, psif, pblk] = nkp_block_precond_running(phi, phif, Kblk, Bblk, pblk, alpha_p, eps_n)
    % running block-power preconditioning
    D = numel(phi);
    psi  = phi;
    psif = phif;

    for k = 1:Kblk
        i1 = (k-1)*Bblk + 1;
        if k < Kblk
            i2 = k*Bblk;
        else
            i2 = D;
        end

        blkf = phif(i1:i2);

        pk_inst = mean(blkf.^2);
        pblk(k) = alpha_p*pblk(k) + (1-alpha_p)*pk_inst;

        scale = 1/sqrt(pblk(k) + eps_n);

        psi(i1:i2)  = phi(i1:i2)  * scale;
        psif(i1:i2) = phif(i1:i2) * scale;
    end
end
