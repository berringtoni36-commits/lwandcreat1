clear; close all; clc;

%% ---------------- Global settings ----------------
T       = 30000;
inde    = 20;
fg      = 0.9995;
eps_n   = 1e-10;
eps_anr = 1e-12;

rng(2026,'twister');

%% ---------------- Primary / Secondary paths ----------------
Pw = [0 0 0 1 -0.3 0.2];
Sw = [0 0 1 0.5];
Lp = length(Pw);
Ls = length(Sw);

%% ---------------- RFF+FxNLMS settings ----------------
M       = 20;
mu_nlms = 0.04;

%% ---------------- MCC settings ----------------
mu_mcc    = 0.04;
sigma_mcc = 2.0;

%% ---------------- NKP settings ----------------
mu_nkp    = 0.06;
Kblk      = 10;
pblk      = 0.7;
alpha_p   = 0.9;
scale_max = 3.0;

%% ---------------- RFF settings ----------------
D         = 500;
sigma_rff = 5;

Omega = randn(D,M) / sigma_rff;
bb    = 2*pi*rand(D,1);

%% ---------------- Storage ----------------
ANR1 = zeros(inde,T);   % RFF+FxNLMS
ANR2 = zeros(inde,T);   % RFF+FxNLMS+MCC
ANR3 = zeros(inde,T);   % NKP+RFF+FxNLMS+MCC
Xsig = zeros(inde,T);

%% ---------------- Monte Carlo loop ----------------
for mc = 1:inde
    fprintf('Monte Carlo run %d / %d\n', mc, inde);

    x = randn(T,1);
    d = filter(Pw, 1, x);
    xf_seq = filter(Sw, 1, x);

    xbuf  = zeros(M,1);
    xfbuf = zeros(M,1);

    ybuf1 = zeros(Ls,1);
    ybuf2 = zeros(Ls,1);
    ybuf3 = zeros(Ls,1);

    w1 = zeros(D,1);
    w2 = zeros(D,1);
    w3 = zeros(D,1);

    Pe1 = 0; Pd1 = 0;
    Pe2 = 0; Pd2 = 0;
    Pe3 = 0; Pd3 = 0;

    for n = 1:T
        xbuf  = [x(n);      xbuf(1:end-1)];
        xfbuf = [xf_seq(n); xfbuf(1:end-1)];

        xvec  = xbuf;
        xfvec = xfbuf;

        phi  = sqrt(2/D) * cos(Omega*xvec  + bb);
        phif = sqrt(2/D) * cos(Omega*xfvec + bb);

        % common measurement noise for fair comparison
        v = 0.01*randn;

        % ----- Algorithm 1: RFF+FxNLMS -----
        y1 = w1.' * phi;
        ybuf1 = [y1; ybuf1(1:end-1)];
        ys1 = Sw * ybuf1;

        e1 = d(n) - ys1 + v;

        norm1 = phif.' * phif + eps_n;
        w1 = w1 + mu_nlms * (e1 / norm1) * phif;

        Pd1 = fg * Pd1 + (1-fg) * (d(n)^2);
        Pe1 = fg * Pe1 + (1-fg) * (e1^2);

        if n <= Lp
            ANR1(mc,n) = 0;
        else
            ANR1(mc,n) = 20*log10((Pe1+eps_anr)/(Pd1+eps_anr));
        end

        % ----- Algorithm 2: RFF+FxNLMS+MCC -----
        y2 = w2.' * phi;
        ybuf2 = [y2; ybuf2(1:end-1)];
        ys2 = Sw * ybuf2;

        e2 = d(n) - ys2 + v;

        g2 = exp(-(e2^2)/(2*sigma_mcc^2));

        norm2 = phif.' * phif + eps_n;
        w2 = w2 + mu_mcc * g2 * (e2 / norm2) * phif;

        Pd2 = fg * Pd2 + (1-fg) * (d(n)^2);
        Pe2 = fg * Pe2 + (1-fg) * (e2^2);

        if n <= Lp
            ANR2(mc,n) = 0;
        else
            ANR2(mc,n) = 20*log10((Pe2+eps_anr)/(Pd2+eps_anr));
        end

        % ----- Algorithm 3: NKP+RFF+FxNLMS+MCC -----
        y3 = w3.' * phi;
        ybuf3 = [y3; ybuf3(1:end-1)];
        ys3 = Sw * ybuf3;

        e3 = d(n) - ys3 + v;

        g3 = exp(-(e3^2)/(2*sigma_mcc^2));

        phif_nkp = nkp_feature_only(phif, Kblk, pblk, alpha_p, eps_n, scale_max);

        norm3 = phif_nkp.' * phif_nkp + eps_n;
        w3 = w3 + mu_nkp * g3 * (e3 / norm3) * phif_nkp;

        Pd3 = fg * Pd3 + (1-fg) * (d(n)^2);
        Pe3 = fg * Pe3 + (1-fg) * (e3^2);

        if n <= Lp
            ANR3(mc,n) = 0;
        else
            ANR3(mc,n) = 20*log10((Pe3+eps_anr)/(Pd3+eps_anr));
        end

        Xsig(mc,n) = x(n);
    end
end

%% ---------------- Average ANR ----------------
ANR1_mean = mean(ANR1,1);
ANR2_mean = mean(ANR2,1);
ANR3_mean = mean(ANR3,1);
X_mean    = mean(Xsig,1);

ANR_all = [ANR1_mean; ANR2_mean; ANR3_mean];

%% ---------------- Plot ----------------
figure('Position', [100, 100, 500, 400]);

colorc = ['-.'; '--'; '- '; '-D'; ': '; ': '; ': '; '-.'; '--'; '- '; '-^'];
linewd = 1.6;
FontSize = 15;
aa = 1:1:T;

for index = 1:3
    plot(aa(Lp+1:end), ANR_all(index,Lp+1:end), colorc(index,:), ...
        'Markersize',5,'Linewidth',linewd);
    hold on;
end
grid on;

set(gca,'LineWidth',linewd);
xlabel('Iteration','fontsize',FontSize,'interpreter', 'latex');
ylabel('ANR(dB)','fontsize',FontSize,'interpreter', 'latex');
set(gca,'fontsize',FontSize,'fontname','Times new roman');

xlim([0 30000]);
ylim([-15 -8]);

set(gca,'XTick',[0 5000 10000 15000 20000 25000 30000]);
set(gca,'YTick',[-45 -40 -35 -30 -25 -20 -15 -10 -5 0]);

h = legend({'RFF+NLMS','RFF+NLMS+MCC','NKP+RFF+NLMS+MCC'}, ...
    'fontsize',FontSize-1,'Location','North','Interpreter','latex');
set(h,'FontSize',FontSize,'FontName','Times New Roman');
set(gca,'Fontname','times new Roman');

function vout = nkp_feature_only(vin, Kblk, pblk, alpha_p, eps_n, scale_max)
    D = length(vin);
    vout = zeros(size(vin));

    nb = ceil(D / Kblk);

    for ib = 1:nb
        st = (ib-1)*Kblk + 1;
        ed = min(ib*Kblk, D);

        blk = vin(st:ed);

        s = norm(blk) + eps_n;
        z = blk / s;

        z = sign(z) .* (abs(z).^alpha_p);
        z = pblk * z;

        z(z >  scale_max) =  scale_max;
        z(z < -scale_max) = -scale_max;

        vout(st:ed) = s * z;
    end
end