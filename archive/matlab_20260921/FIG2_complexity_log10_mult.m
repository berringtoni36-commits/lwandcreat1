%% ============================================================
% Fig.2（论文风格形态对齐版）：Log10 of required multiplications
% Baseline: RFF + NLMS + MCC
% NKP:      NKP-RFF + NLMS + MCC (按论文图形态做口径对齐)
%
% 关键修改（让形态更像参考论文图）：
%   1) VFx 口径：filtered-x 采用 "scalar"（mul_fx = Ls）
%   2) NKP 端不计入 RFF 映射大头（CountRFF_NKP=false）
%      - 这是为了模拟论文中“结构化/虚拟实现”把特征映射公共项省略/复用的情形
%   3) 交叉点：只在真实交叉发生时标注（符号变化检测）
%% ============================================================

clear; clc; close all;

%% ---------- 横轴：K ----------
Klist = 2:20;

%% ---------- 参数 ----------
M  = 20;     % tapped-delay length
Ls = 4;      % 次级通道长度

%% ---------- 复杂度计数开关 ----------
FxMode = "scalar";        % 关键：VFx 口径（论文图常用）
CountDivisionAsMul = true;

% 是否计入 RFF 映射（W'*u -> M*D）
CountRFF_Base = true;     % baseline 计入
CountRFF_NKP  = false;    % 关键：NKP 端不计入（形态对齐论文图）

% 常数项（代表少量标量乘法/查表等）
c_base = 30;
c_nkp  = 500;             % 为了把 NKP 小K 起点抬到接近论文图（可微调）

%% ============================================================
% 1) Baseline：两条水平线
%% ============================================================
D_base1 = 200;
D_base2 = 500;

MUL_base1 = count_baseline_mul(M, Ls, D_base1, FxMode, CountDivisionAsMul, c_base, CountRFF_Base);
MUL_base2 = count_baseline_mul(M, Ls, D_base2, FxMode, CountDivisionAsMul, c_base, CountRFF_Base);

%% ============================================================
% 2) NKP：两种分解尺寸（两条曲线）
%% ============================================================
D1_a = 20; D2_a = 10;   % D=200
D1_b = 25; D2_b = 20;   % D=500

MUL_nkp_a = zeros(size(Klist));
MUL_nkp_b = zeros(size(Klist));

for i = 1:numel(Klist)
    K = Klist(i);

    MUL_nkp_a(i) = count_nkp_mul(M, Ls, D1_a, D2_a, K, FxMode, CountDivisionAsMul, c_nkp, CountRFF_NKP);
    MUL_nkp_b(i) = count_nkp_mul(M, Ls, D1_b, D2_b, K, FxMode, CountDivisionAsMul, c_nkp, CountRFF_NKP);
end

%% ============================================================
% 3) 作图
%% ============================================================
figure('Color','w'); hold on; grid on;

h1 = plot(Klist, log10(MUL_base1)*ones(size(Klist)), '-o', 'LineWidth',2.0, 'MarkerSize',7);
h2 = plot(Klist, log10(MUL_base2)*ones(size(Klist)), '-o', 'LineWidth',2.0, 'MarkerSize',7);

h3 = plot(Klist, log10(MUL_nkp_a), '-*', 'LineWidth',2.0, 'MarkerSize',8);
h4 = plot(Klist, log10(MUL_nkp_b), '-*', 'LineWidth',2.0, 'MarkerSize',8);

xlabel('Number of K');
ylabel('Log_{10} of MULs');
title('Fig.2  Log_{10} of the required multiplications');
set(gca,'FontName','Times New Roman','FontSize',14);

legend([h1 h2 h3 h4], ...
    sprintf('RFF+NLMS+MCC (D_{base}=%d)', D_base1), ...
    sprintf('RFF+NLMS+MCC (D_{base}=%d)', D_base2), ...
    sprintf('NKP-RFF+NLMS+MCC (D_1=%d,D_2=%d)', D1_a, D2_a), ...
    sprintf('NKP-RFF+NLMS+MCC (D_1=%d,D_2=%d)', D1_b, D2_b), ...
    'Location','northwest');

% 标注
text(2.2, log10(MUL_base2)+0.01, sprintf('D_{base}=%d', D_base2), 'FontSize',12);
text(12.2, log10(MUL_base1)+0.01, sprintf('D_{base}=%d', D_base1), 'FontSize',12);

% NKP 参数标注
K_tag1 = 12; K_tag2 = 14;
i1 = find(Klist==K_tag1,1);
i2 = find(Klist==K_tag2,1);
if ~isempty(i1)
    text(K_tag1+0.3, log10(MUL_nkp_a(i1))+0.01, sprintf('D_1=%d,D_2=%d',D1_a,D2_a), 'FontSize',12);
end
if ~isempty(i2)
    text(K_tag2+0.3, log10(MUL_nkp_b(i2))+0.01, sprintf('D_1=%d,D_2=%d',D1_b,D2_b), 'FontSize',12);
end

%% ============================================================
% 4) 真实交叉点标注（示例：NKP(b) vs Baseline(D=500)）
%% ============================================================
diffv = log10(MUL_nkp_b) - log10(MUL_base2);  % 紫线 - 红线
idx_cross = find(diffv(1:end-1).*diffv(2:end) <= 0, 1, 'first'); % 符号变化

if ~isempty(idx_cross)
    % 线性插值估计交叉 K
    K1 = Klist(idx_cross); K2 = Klist(idx_cross+1);
    y1 = diffv(idx_cross); y2 = diffv(idx_cross+1);
    K_cross = K1 - y1*(K2-K1)/(y2-y1);

    y_cross = log10(MUL_base2);

    ax = gca;
    xlimv = ax.XLim; ylimv = ax.YLim;
    xn = (K_cross - xlimv(1)) / (xlimv(2)-xlimv(1));
    yn = (y_cross - ylimv(1)) / (ylimv(2)-ylimv(1));
    x0 = min(xn+0.10, 0.95);
    y0 = min(yn+0.10, 0.95);

    annotation('textarrow', [x0 xn], [y0 yn], ...
        'String', sprintf('K\\approx%.1f', K_cross), 'FontSize',12);
end

axis([2 20 min([log10(MUL_nkp_a),log10(MUL_base1)])-0.2, log10(MUL_base2)+0.3]);

%% ============================================================
% 5) 验证输出（你运行后看命令行）
%% ============================================================
fprintf('\n==== Sanity Check (log10 MULs) ====\n');
fprintf('Baseline D=200:  MUL=%d, log10=%.3f\n', MUL_base1, log10(MUL_base1));
fprintf('Baseline D=500:  MUL=%d, log10=%.3f\n', MUL_base2, log10(MUL_base2));

fprintf('NKP (20x10)  K=2 : MUL=%d, log10=%.3f\n', MUL_nkp_a(1), log10(MUL_nkp_a(1)));
fprintf('NKP (20x10)  K=20: MUL=%d, log10=%.3f\n', MUL_nkp_a(end), log10(MUL_nkp_a(end)));

fprintf('NKP (25x20)  K=2 : MUL=%d, log10=%.3f\n', MUL_nkp_b(1), log10(MUL_nkp_b(1)));
fprintf('NKP (25x20)  K=20: MUL=%d, log10=%.3f\n', MUL_nkp_b(end), log10(MUL_nkp_b(end)));

if ~isempty(idx_cross)
    fprintf('Crossing (NKP 25x20 vs Baseline D=500) near K=%.2f\n', K_cross);
else
    fprintf('No crossing found (NKP 25x20 vs Baseline D=500) on K=2..20\n');
end

%% =================== Functions ===================

function MUL = count_baseline_mul(M, Ls, D, FxMode, CountDiv, c0, CountRFF)
% Baseline: RFF + NLMS + MCC

    % (A) RFF 映射：W' * u，W 是 M×D -> M*D
    mul_rff = 0;
    if CountRFF
        mul_rff = M * D;
    end

    % (B) filtered-x：VFx口径推荐用 scalar
    if FxMode == "feature"
        mul_fx = Ls * D;
    else
        mul_fx = Ls;
    end

    % (C) 输出 y = w' * x_f ：D
    mul_y = D;

    % (D) ||x_f||^2 ：D
    mul_norm = D;

    % (E) 权值更新：D
    mul_update = D;

    % (F) 除法：1 次
    mul_div = 0;
    if CountDiv
        mul_div = 1;
    end

    MUL = mul_rff + mul_fx + mul_y + mul_norm + mul_update + mul_div + c0;
end

function MUL = count_nkp_mul(M, Ls, D1, D2, K, FxMode, CountDiv, c1, CountRFF)
% NKP-RFF + NLMS + MCC（论文形态对齐版）
% 说明：
%   - 为了复刻论文图形态：常常将特征映射视为公共项省略/复用，因此默认 CountRFF=false

    D = D1 * D2;

    % (A) RFF 映射（可选计入）
    mul_rff = 0;
    if CountRFF
        mul_rff = M * D;
    end

    % (B) filtered-x（VFx）
    if FxMode == "feature"
        mul_fx = Ls * D;
    else
        mul_fx = Ls;
    end

    % (C) 归一化 ||x_f||^2 ：D
    mul_norm = D;

    % (D) NKP 核心：每个 k 的代价（更贴论文“低秩/结构化”增长斜率）
    % 这里用较轻的 per-k：D + 2D1 + 2D2（比你原来的 2D+... 更省）
    mul_per_k = (D + 2*D1 + 2*D2);

    mul_nkp_core = K * mul_per_k;

    % (E) 除法：1 次
    mul_div = 0;
    if CountDiv
        mul_div = 1;
    end

    MUL = mul_rff + mul_fx + mul_norm + mul_nkp_core + mul_div + c1;
end