% ============================================================
% FIG6_Logistic_RFF_MCC_NKP.m
% Fig.6 Logistic chaotic noise source (no inset)
% Curves: RFF+NLMS+MCC (baseline), NKP+NLMS+MCC (proposed)
% ============================================================

clear; close all; clc;

%% ---------------- Global settings ----------------
T       = 20000;        % iterations (x-axis shows 0~2 x 10^4)
inde    = 20;            % Monte Carlo trials
M       = 20;           % tapped-delay length (input vector length)
fg      = 0.999;        % ANR smoothing factor
eps_n   = 1e-10;
eps_anr = 1e-12;

rng(871,'twister');

%% ---------------- Paths (given by you) ----------------
Pw = [0 0 0 1 -0.3 0.2];   % primary path
Sw = [0 0 1 0.5];          % secondary path (minimum-phase)
Ls = numel(Sw);

%% ---------------- RFF / MCC / NKP params ----------------
D      = 500;       % RFF feature dimension
sigma  = 2.0;       % MCC kernel width
kappa  = 2;         % MCC scale
mu_mcc = 0.7;      % step-size for RFF+NLMS+MCC
mu_nkp = 0.5;      % step-size for NKP+NLMS+MCC

% NKP block preconditioning (paper-aligned)
Kblk   = 4;                      % paper Fig.6 uses K=4
Bblk   = floor(D / Kblk);

alpha_p = 0.99;                  % running block-power smoothing (0.98~0.995)

%% ---------------- Random Fourier Features ----------------
gamma_rff = 1.0;
W = gamma_rff * randn(D, M);
b = 2*pi*rand(D, 1);
rff_map = @(xvec) sqrt(2/D) * cos(W*xvec + b);

%% ---------------- Allocate ----------------
ANR_mcc_acc  = zeros(T,1);
ANR_nkp_acc  = zeros(T,1);

%% ================= Monte Carlo =================
for it = 1:inde
pblk = ones(Kblk,1);   % running block power init (NKP only)

   % ==========================================================
% (1) Logistic chaotic input (paper): u(n)=4*u(n-6)*(1-u(n-6))
% ==========================================================
u = logistic_delay6(T, 0.37);     % x0=0.37 (paper-like)
u = u - mean(u);
u = u / (std(u) + 1e-12);

% ==========================================================
% (2) Nonlinear disturbance (paper):
% g = P(z)*u
% d(n)=g(n-2)+0.08*g^2(n-2)-0.04*g^3(n-1)
% ==========================================================
g = filter(Pw, 1, u);

d = zeros(T,1);
n0 = 3:T;   % to avoid negative indices
d(n0) = g(n0-2) + 0.08*(g(n0-2).^2) - 0.04*(g(n0-1).^3);

% filtered reference through secondary path
xf = filter(Sw, 1, u);

% 后面你的 xbuf / xfbuf 输入也要对应改成 u(n)
% 即：xbuf = [u(n); xbuf(1:end-1)];


    % ----- Initialize weights -----
    w_mcc  = zeros(D,1);
    w_nkp  = zeros(D,1);

    % buffers for tapped-delay vectors
    xbuf  = zeros(M,1);
    xfbuf = zeros(M,1);

    % buffer for secondary path output (scalar y -> y_s)
    ybuf_mcc  = zeros(Ls,1);
    ybuf_nkp  = zeros(Ls,1);

    % ANR smoothing
    Pd_mcc  = 0; Pe_mcc  = 0;
    Pd_nkp  = 0; Pe_nkp  = 0;

    ANR_mcc  = zeros(T,1);
    ANR_nkp  = zeros(T,1);

    for n = 1:T
        % update buffers
        xbuf  = [u(n);  xbuf(1:end-1)];
        xfbuf = [xf(n); xfbuf(1:end-1)];

        % RFF features
        phi  = rff_map(xbuf);
        phif = rff_map(xfbuf);

        % ================= RFF + NLMS + MCC (baseline) =================
        y_mcc = w_mcc' * phi;
        ybuf_mcc = [y_mcc; ybuf_mcc(1:end-1)];
        ys_mcc = Sw * ybuf_mcc;
        e_mcc  = d(n) - ys_mcc;

        g = kappa * exp( - (e_mcc^2) / (2*sigma^2) );  % MCC weight
        den = (phif'*phif + eps_n);
        w_mcc = w_mcc + mu_mcc * g * (e_mcc/den) * phif;

        Pd_mcc = fg*Pd_mcc + (1-fg)*(d(n)^2);
        Pe_mcc = fg*Pe_mcc + (1-fg)*(e_mcc^2);
        ANR_mcc(n) = 10*log10( (Pd_mcc+eps_anr)/(Pe_mcc+eps_anr) );

        % ================= NKP + NLMS + MCC (proposed) =================
       [psi, psif, pblk] = nkp_block_precond_running(phi, phif, Kblk, Bblk, pblk, alpha_p, eps_n);

        y_nkp = w_nkp' * psi;
        ybuf_nkp = [y_nkp; ybuf_nkp(1:end-1)];
        ys_nkp = Sw * ybuf_nkp;
        e_nkp  = d(n) - ys_nkp;

        g = kappa * exp( - (e_nkp^2) / (2*sigma^2) );
        den = (psif'*psif + eps_n);
        w_nkp = w_nkp + mu_nkp * g * (e_nkp/den) * psif;

        Pd_nkp = fg*Pd_nkp + (1-fg)*(d(n)^2);
        Pe_nkp = fg*Pe_nkp + (1-fg)*(e_nkp^2);
        ANR_nkp(n) = 10*log10( (Pd_nkp+eps_anr)/(Pe_nkp+eps_anr) );
    end

    % accumulate
    ANR_mcc_acc  = ANR_mcc_acc  + ANR_mcc;
    ANR_nkp_acc  = ANR_nkp_acc  + ANR_nkp;
end

ANR_mcc  = ANR_mcc_acc / inde;
ANR_nkp  = ANR_nkp_acc / inde;

ANR_mcc(~isfinite(ANR_mcc)) = NaN;
ANR_nkp(~isfinite(ANR_nkp)) = NaN;

% ---------------- Plot (paper-like) ----------------
t  = 0:T-1;                  % 直接用迭代次数，不要除以 1e4
mk = round(linspace(1,T,11));

figure('Color',[0.94 0.94 0.94]); hold on; grid on; box on;
set(gca,'Color',[0.94 0.94 0.94],'FontName','Times New Roman',...
    'FontSize',18,'LineWidth',1.2);

plot(t, ANR_mcc, 'LineWidth',2.4, 'Marker','o', 'MarkerIndices',mk,...
    'MarkerSize',10, 'DisplayName','RFF+NLMS+MCC (baseline)');
plot(t, ANR_nkp, 'LineWidth',2.6, 'Marker','^', 'MarkerIndices',mk,...
    'MarkerSize',12, 'DisplayName','NKP+NLMS+MCC (proposed)');

xlabel('Iteration','FontName','Times New Roman','FontSize',22);
ylabel('ANR (dB)','FontName','Times New Roman','FontSize',22);
title('Fig.6  Logistic chaotic noise source','FontName','Times New Roman','FontSize',24);

xlim([0 2e4]);               % 0~20000
ylim([-35 2]);

ax = gca;
ax.XAxis.Exponent = 4;       % 右下角显示 ×10^4（此时才是正确的）
legend('Location','northeast');


%% ================= helper functions =================
function x = logistic_spiky(T, r, x0, thr)
    x = zeros(T,1);
    z = x0;
    for n = 1:T
        z = r*z*(1-z);
        if z > thr
            x(n) = z;
        else
            x(n) = 0;
        end
    end
end

function u = logistic_delay6(T, u0)
    % paper: u(n)=4*u(n-6)*(1-u(n-6))
    u = zeros(T,1);
    u(1:6) = u0;
    for n = 7:T
        u(n) = 4*u(n-6)*(1-u(n-6));
    end
end


function [psi, psif, pblk] = nkp_block_precond_running(phi, phif, Kblk, Bblk, pblk, alpha_p, eps_n)
% NKP: running block-power preconditioning (paper-like)
% 用 filtered feature (phif) 的 block 能量做平滑估计，再对 phi/phif 同步预条件化

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

        % running power estimate
        pk_inst = mean(blkf.^2);
        pblk(k) = alpha_p*pblk(k) + (1-alpha_p)*pk_inst;

        scale = 1/sqrt(pblk(k) + eps_n);

        psi(i1:i2)  = phi(i1:i2)  * scale;
        psif(i1:i2) = phif(i1:i2) * scale;
    end
end