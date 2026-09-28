%% ============================================================
% 论文同款 Fig.(a)(b)：幅频 + 相频（左右子图），并导出高清图
% 关键：不要截图，用 exportgraphics 防止标题被挡
%
% 输出：Fig3_MagPhase_likePaper.png (300 dpi)
%% ============================================================

clear; clc; close all;

%% ---------- 通道 ----------
Pz = [0 0 0 1 -0.3 0.2];
S_mp  = [0 0 1 0.5];
S_nmp = [0 0 1 1.5 -1];

sec_mode = 1;   % 1: minimum-phase  2: non-minimum-phase
if sec_mode==1
    Sz = S_mp;
else
    Sz = S_nmp;
end

%% ---------- 频响 ----------
Nfft = 2048;
[Hp,w] = freqz(Pz,1,Nfft);
[Hs,~] = freqz(Sz,1,Nfft);

x = w/pi;  % Normalized Frequency (×π rad/sample)

Hp_mag = 20*log10(abs(Hp)+1e-12);
Hs_mag = 20*log10(abs(Hs)+1e-12);

Hp_ph = unwrap(angle(Hp))*180/pi;
Hs_ph = unwrap(angle(Hs))*180/pi;

%% ---------- 外观设置（贴论文） ----------
fig = figure('Color',[0.92 0.92 0.92]);              % 灰底
set(fig,'Position',[120 120 1250 520]);              % 画布比例像论文

tl = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');

% ========== (a) 幅频 ==========
ax1 = nexttile; hold(ax1,'on'); box(ax1,'on'); grid(ax1,'on');
set(ax1,'Color','w');                                  % 坐标轴白底更像论文
plot(ax1, x, Hp_mag, 'LineWidth', 2.8);
plot(ax1, x, Hs_mag, 'LineWidth', 2.8);
xlabel(ax1,'Normalized Frequency (\times\pi rad/sample)');
ylabel(ax1,'Magnitude (dB)');
legend(ax1,'P(z)','S(z)','Location','northeast');
title(ax1,'(a)  Magnitude-frequency characteristic.','FontWeight','normal');
set(ax1,'FontName','Times New Roman','FontSize',18,'LineWidth',1.2);
xlim(ax1,[0 1]);

% ========== (b) 相频 ==========
ax2 = nexttile; hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');
set(ax2,'Color','w');
plot(ax2, x, Hp_ph, 'LineWidth', 2.8);
plot(ax2, x, Hs_ph, 'LineWidth', 2.8);
xlabel(ax2,'Normalized Frequency (\times\pi rad/sample)');
ylabel(ax2,'Phase (degrees)');
legend(ax2,'P(z)','S(z)','Location','northeast');
title(ax2,'(b)  Phase-frequency characteristic.','FontWeight','normal');
set(ax2,'FontName','Times New Roman','FontSize',18,'LineWidth',1.2);
xlim(ax2,[0 1]);
ylim(ax2,[-600 0]);     % 固定相位范围，更像论文

% 让两张子图顶部留白，避免标题靠太上
tl.Padding = 'compact';
tl.TileSpacing = 'compact';

%% ---------- 导出高清图（避免标题被工具栏遮挡） ----------
exportgraphics(fig,'Fig3_MagPhase_likePaper.png','Resolution',300);
disp('Saved: Fig3_MagPhase_likePaper.png (300 dpi)');
