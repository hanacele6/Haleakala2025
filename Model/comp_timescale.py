# -*- coding: utf-8 -*-
"""
【スライダー連動版】水星表面ナトリウムのタイムスケール比較（天頂角 cos_theta 可変型）
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import japanize_matplotlib

# ==============================================================================
# 物理定数・シミュレーション設定（本体コードと完全同期）
# ==============================================================================
N_LON = 72                 # 経度分割数
CELL_DEG = 360.0 / N_LON   # 1セルの経度幅 [deg] (5度)
CELL_RAD = np.deg2rad(CELL_DEG)

# 水星の軌道・自転定数
GM_SUN = 6.6743e-11 * 1.989e30          # G * M_sun [m^3/s^2]
AU_METERS = 1.496e11                     # 1 AU [m]
A_MERCURY = 0.387098 * AU_METERS         # 軌道長半径 [m]
E_MERCURY = 0.205630                     # 離心率
ROTATION_PERIOD = 58.6462 * 86400        # 自転周期 [s]
W_ROT = 2.0 * np.pi / ROTATION_PERIOD    # 自転角速度 [rad/s]

# 放出物理定数
NU_0 = 1.0e13              # TDの頻度因子 [1/s]
K_B_EV = 8.617e-5          # ボルツマン定数 [eV/K]
TEMP_BASE = 100.0          # 表面温度のベース [K]
TEMP_AMP = 600.0           # 表面温度の振幅 [K]

Q_PSD_BASE = 2.0           # Q_PSDのベース係数
Q_PSD_M2 = (Q_PSD_BASE * 1.0e-20) / (100 ** 2)  # [m^2]
F_UV_1AU_M2 = 1.5e14 * (100 ** 2)               # [photons/m^2/s]

# 注目する束縛エネルギー U [eV]
U_EV = 1.85

# ==============================================================================
# 共通データの事前計算
# ==============================================================================
taa_deg = np.linspace(0.0, 360.0, 500)
taa_rad = np.deg2rad(taa_deg)

# 理論的な公転角速度 omega_orb
n_motion = np.sqrt(GM_SUN / (A_MERCURY ** 3))
w_orb = n_motion * ((1.0 + E_MERCURY * np.cos(taa_rad)) ** 2) / ((1.0 - E_MERCURY ** 2) ** 1.5)

# 見かけの直下点移動速度とセル移動タイムスケール (cos_theta に依存しない一意の物差し)
w_sun_apparent = np.abs(W_ROT - w_orb)
tau_rot = np.where(w_sun_apparent > 1e-20, CELL_RAD / w_sun_apparent, 1e15)

# 太陽・水星間距離 R [AU]
au_dist = 0.387098 * (1.0 - E_MERCURY ** 2) / (1.0 + E_MERCURY * np.cos(taa_rad))

# ==============================================================================
# 初期描画のセットアップ
# ==============================================================================
# ウィンドウの作成と配置の調整（下にスライダーのスペースを確保）
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.20) 

# ① 黒実線：セル移動タイムスケール（固定）
line_rot, = ax.plot(taa_deg, tau_rot, color='black', linestyle='-', linewidth=2.5, 
                     label='セル移動 (自転移流)')

# 初期値の計算 (cos_theta = 1.0)
initial_cos = 1.0
t_surf = TEMP_BASE + TEMP_AMP * (initial_cos ** 0.25) * np.sqrt(0.306 / au_dist)
r_td = NU_0 * np.exp(-U_EV / (K_B_EV * t_surf))
r_psd = (F_UV_1AU_M2 / (au_dist ** 2)) * Q_PSD_M2 * initial_cos
r_max = np.maximum(r_psd, r_td)

# ② 赤破線：実際の放出タイムスケール（可変）
line_release, = ax.plot(taa_deg, 1.0 / r_max, color='crimson', linestyle='--', linewidth=2.0, 
                        label='放出タイムスケール (PSD+TD速い方)')

# ③ 青点線：PSD単体限界線（可変）
line_psd, = ax.plot(taa_deg, 1.0 / r_psd, color='royalblue', linestyle=':', linewidth=1.5, 
                    alpha=0.8, label='PSD単体限界線')

# グラフの静的修飾
ax.set_yscale('log')
ax.set_xlim(0, 360)
ax.set_ylim(1e-1, 1e9)
ax.set_xlabel('真近点角 (TAA) [deg]', fontsize=12, fontweight='bold')
ax.set_ylabel('タイムスケール [秒]', fontsize=12, fontweight='bold')
ax.grid(True, which='both', linestyle='--', alpha=0.4)

# 太陽の停止・逆行域の縦線
ax.axvline(19.5, color='gray', linestyle='-', alpha=0.3)
ax.axvline(340.5, color='gray', linestyle='-', alpha=0.3)

# 時間の目安ライン
ax.axhline(3600, color='gray', linestyle=':', alpha=0.5)
ax.text(355, 3600 * 1.2, '1 時間', color='gray', ha='right', va='bottom')
ax.axhline(86400, color='gray', linestyle=':', alpha=0.5)
ax.text(355, 86400 * 1.2, '1 日', color='gray', ha='right', va='bottom')

# 動的タイトルの初期設定
title_obj = ax.set_title(f'水星ナトリウム タイムスケール比較 (天頂角 cos\u03b8 = {initial_cos:.2f}, U = {U_EV} eV)', 
                         fontsize=14, fontweight='bold')

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# ==============================================================================
# スライダーの設定と更新イベント
# ==============================================================================
# スライダー用の描画領域 [左, 下, 幅, 高さ]
ax_slider = plt.axes([0.15, 0.07, 0.70, 0.04], facecolor='lightgray')
slider_cos = Slider(
    ax_slider, 
    label='天頂角 cos\u03b8 ', 
    valmin=0.05,        # 完全にゼロだとPSDも無限に発散するので 0.05 を下限に
    valmax=1.0, 
    valinit=initial_cos, 
    valfmt='%.2f',
    color='crimson'
)

# スライダーが動いたときに呼び出される関数
def update(val):
    current_cos = slider_cos.val
    
    # 新しい物理量の再計算
    t_surf_new = TEMP_BASE + TEMP_AMP * (current_cos ** 0.25) * np.sqrt(0.306 / au_dist)
    r_td_new = NU_0 * np.exp(-U_EV / (K_B_EV * t_surf_new))
    r_psd_new = (F_UV_1AU_M2 / (au_dist ** 2)) * Q_PSD_M2 * current_cos
    r_max_new = np.maximum(r_psd_new, r_td_new)
    
    # 線のデータを更新（高速リドロー）
    line_release.set_ydata(1.0 / r_max_new)
    line_psd.set_ydata(1.0 / r_psd_new)
    
    # タイトルのテキストも天頂角に合わせて書き換え
    title_obj.set_text(f'水星ナトリウム タイムスケール比較 (天頂角 cos\u03b8 = {current_cos:.2f}, U = {U_EV} eV)')
    
    # キャンバスの再描画
    fig.canvas.draw_idle()

# イベントの紐付け
slider_cos.on_changed(update)

print("スライダー付きウィンドウを起動しました。")
plt.show()