%% ============================================================
% Fig.4：Logistic chaotic noise source（尖峰型，贴论文风格）
% 核心：用标准 Logistic + 幂变换 x=u^p 产生“接近0+偶发尖峰”
%% ============================================================

clear; clc; close all;

%% ---------- 参数 ----------
mu = 4.0;           % Logistic 参数
u0 = 0.37;          % 初值（避免 0/0.25/0.5/0.75/1）
burn_in = 2000;     % 丢弃前面过渡
T_show = 200;       % 只画 0~200
p = 10;             % 幂指数：越大越“尖峰”(建议 8~12)

T_gen = burn_in + T_show + 10;

%% ---------- 生成标准 Logistic 序列（无延迟） ----------
u = zeros(T_gen,1);
u(1) = u0;
for n = 2:T_gen
    u(n) = mu * u(n-1) * (1 - u(n-1));
end

u_use = u(burn_in+1 : burn_in+T_show);

% 幂变换：制造“尖峰型”外观（贴论文）
x = u_use.^p;

%% ---------- 作图（灰底+大字号+底部caption） ----------
fig = figure('Color',[0.92 0.92 0.92]);
set(fig,'Position',[180 180 1100 420]);

ax = axes('Parent',fig);
set(ax,'Color',[0.92 0.92 0.92]);
hold on; box on; grid on;

plot(0:T_show-1, x, 'LineWidth', 2.4);

xlabel('Iteration');
ylabel('Amplitude');
xlim([0 T_show]);
ylim([0 1]);

set(gca,'FontName','Times New Roman','FontSize',18,'LineWidth',1.2);

% 底部图注（不要和 x-label 打架）
annotation('textbox',[0 0.01 1 0.10],...
    'String','Fig. 4.  Logistic chaotic noise source.',...
    'EdgeColor','none','HorizontalAlignment','center',...
    'FontName','Times New Roman','FontSize',22,'FontWeight','normal');

exportgraphics(fig,'Fig4_Logistic_spiky_likePaper.png','Resolution',300);
disp('Saved: Fig4_Logistic_spiky_likePaper.png');
