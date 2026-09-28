clear; close all; clc;

%% ================= Global settings =================
T       = 30000;      % number of iterations
inde    = 20;         % Monte Carlo runs
fg      = 0.9995;     % forgetting factor for ANR
eps_n   = 1e-10;
eps_anr = 1e-12;

rng(2026,'twister');

%% ================= Primary / Secondary paths =================
Pw = [0 0 0 1 -0.3 0.2];
Sw = [0 0 1 0.5];

Lp = length(Pw);
Ls = length(Sw);

%% ================= RFF+FxNLMS settings =================
M       = 20;         % input tap length
mu_nlms = 0.04;

%% ================= MCC settings =================
mu_mcc    = 0.04;
sigma_mcc = 2.0;

%% ================= NKP settings =================
mu_nkp    = 0.06;
Kblk      = 10;
pblk      = 0.7;
alpha_p   = 0.9;
scale_max = 3.0;

%% ================= RFF settings =================
D         = 500;      % number of random features
sigma_rff = 5;

Omega = randn(D,M) / sigma_rff;
bb    = 2*pi*rand(D,1);

%% ================= alpha-stable noise settings =================
% alpha = 2 -> Gaussian
% alpha < 2 -> impulsive heavy-tailed noise
alpha_ns = 1.4;     % characteristic exponent
beta_ns  = 0.0;     % symmetry parameter
gamma_ns = 0.01;    % scale parameter
delta_ns = 0.0;     % location parameter

%% ================= Storage =================
ANR1 = zeros(inde,T);   % RFF+FxNLMS
ANR2 = zeros(inde,T);   % RFF+FxNLMS+MCC
ANR3 = zeros(inde,T);   % NKP+RFF+FxNLMS+MCC

%% ================= Monte Carlo loop =================
for mc = 1:inde
    fprintf('Monte Carlo run %d / %d\n', mc, inde);

    % input signal
    x = randn(T,1);

    % desired signal through primary path
    d = filter(Pw, 1, x);

    % filtered-x through secondary path
    xf_seq = filter(Sw, 1, x);

    % common alpha-stable measurement noise for fair comparison
    v_seq = alpha_stable_rnd(alpha_ns, beta_ns, gamma_ns, delta_ns, T);

    % buffers
    xbuf  = zeros(M,1);
    xfbuf = zeros(M,1);

    ybuf1 = zeros(Ls,1);
    ybuf2 = zeros(Ls,1);
    ybuf3 = zeros(Ls,1);

    % weights
    w1 = zeros(D,1);
    w2 = zeros(D,1);
    w3 = zeros(D,1);

    % ANR power tracking
    Pe1 = 0; Pd1 = 0;
    Pe2 = 0; Pd2 = 0;
    Pe3 = 0; Pd3 = 0;

    for n = 1:T
        % update input buffers
        xbuf  = [x(n);      xbuf(1:end-1)];
        xfbuf = [xf_seq(n); xfbuf(1:end-1)];

        xvec  = xbuf;
        xfvec = xfbuf;

        % random Fourier feature mapping
        phi  = sqrt(2/D) * cos(Omega*xvec  + bb);
        phif = sqrt(2/D) * cos(Omega*xfvec + bb);

        % common alpha-stable noise sample
        v = v_seq(n);

        %% ----- Algorithm 1: RFF + FxNLMS -----
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
            ANR1(mc,n) = 10*log10((Pe1 + eps_anr) / (Pd1 + eps_anr));
        end

        %% ----- Algorithm 2: RFF + FxNLMS + MCC -----
        y2 = w2.' * phi;
        ybuf2 = [y2; ybuf2(1:end-1)];
        ys2 = Sw * ybuf2;

        e2 = d(n) - ys2 + v;

        g2 = exp(-(e2^2) / (2*sigma_mcc^2));

        norm2 = phif.' * phif + eps_n;
        w2 = w2 + mu_mcc * g2 * (e2 / norm2) * phif;

        Pd2 = fg * Pd2 + (1-fg) * (d(n)^2);
        Pe2 = fg * Pe2 + (1-fg) * (e2^2);

        if n <= Lp
            ANR2(mc,n) = 0;
        else
            ANR2(mc,n) = 10*log10((Pe2 + eps_anr) / (Pd2 + eps_anr));
        end

        %% ----- Algorithm 3: NKP + RFF + FxNLMS + MCC -----
        y3 = w3.' * phi;
        ybuf3 = [y3; ybuf3(1:end-1)];
        ys3 = Sw * ybuf3;

        e3 = d(n) - ys3 + v;

        g3 = exp(-(e3^2) / (2*sigma_mcc^2));

        phif_nkp = nkp_feature_only(phif, Kblk, pblk, alpha_p, eps_n, scale_max);

        norm3 = phif_nkp.' * phif_nkp + eps_n;
        w3 = w3 + mu_nkp * g3 * (e3 / norm3) * phif_nkp;

        Pd3 = fg * Pd3 + (1-fg) * (d(n)^2);
        Pe3 = fg * Pe3 + (1-fg) * (e3^2);

        if n <= Lp
            ANR3(mc,n) = 0;
        else
            ANR3(mc,n) = 10*log10((Pe3 + eps_anr) / (Pd3 + eps_anr));
        end
    end
end

%% ================= Average ANR =================
ANR1_mean = mean(ANR1,1);
ANR2_mean = mean(ANR2,1);
ANR3_mean = mean(ANR3,1);

%% ================= Plot =================
figure('Position', [100, 100, 700, 500]);

aa = 1:T;

plot(aa(Lp+1:end), ANR1_mean(Lp+1:end), '-.', 'LineWidth', 1.6); hold on;
plot(aa(Lp+1:end), ANR2_mean(Lp+1:end), '--', 'LineWidth', 1.6);
plot(aa(Lp+1:end), ANR3_mean(Lp+1:end), '-',  'LineWidth', 1.8);

grid on;
box on;

xlabel('Iteration', 'FontSize', 14);
ylabel('ANR (dB)', 'FontSize', 14);
title(['\alpha-stable noise: \alpha = ', num2str(alpha_ns), ...
       ', \gamma = ', num2str(gamma_ns)], 'FontSize', 13);

legend({'RFF+FxNLMS', 'RFF+FxNLMS+MCC', 'NKP+RFF+FxNLMS+MCC'}, ...
       'Location', 'best', 'FontSize', 12);

set(gca, 'FontSize', 13, 'LineWidth', 1.2);

%% ================= Display final ANR =================
fprintf('\nFinal averaged ANR values:\n');
fprintf('RFF+FxNLMS            : %.4f dB\n', ANR1_mean(end));
fprintf('RFF+FxNLMS+MCC        : %.4f dB\n', ANR2_mean(end));
fprintf('NKP+RFF+FxNLMS+MCC    : %.4f dB\n', ANR3_mean(end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local function 1: NKP feature transform
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Local function 2: alpha-stable random variable generator
%% Chambers-Mallows-Stuck (CMS) method
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = alpha_stable_rnd(alpha, beta, gamma, delta, N)
    % Generate N x 1 alpha-stable random variables
    %
    % alpha in (0, 2]
    % beta  in [-1, 1]
    % gamma > 0
    % delta real
    %
    % For alpha = 2, output reduces to Gaussian-like stable law.

    U = pi * (rand(N,1) - 0.5);      % Uniform(-pi/2, pi/2)
    W = -log(rand(N,1) + eps);       % Exponential(1)

    if abs(alpha - 1) > 1e-12
        B = atan(beta * tan(pi*alpha/2)) / alpha;
        S = (1 + beta^2 * tan(pi*alpha/2)^2)^(1/(2*alpha));

        X0 = S .* sin(alpha*(U + B)) ./ (cos(U).^(1/alpha)) .* ...
            (cos(U - alpha*(U + B)) ./ W).^((1-alpha)/alpha);

        x = gamma * X0 + delta;
    else
        % special case alpha = 1
        X0 = (2/pi) * ( ...
            (pi/2 + beta*U) .* tan(U) - ...
            beta .* log(((pi/2) .* W .* cos(U)) ./ (pi/2 + beta*U)) ...
            );

        x = gamma * X0 + delta;
    end
end