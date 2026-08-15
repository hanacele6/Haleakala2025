# ==============================================================================
# 水星の磁気圏トポロジー 3Dインタラクティブ・シミュレータ
# (Mercury Magnetic Topology 3D Interactive Simulator - Real Tail Edition)
#
# 概要:
#   Lavorenti et al. (2023) に見られるマグネトテイル（尾）を形成するため、
#   カレントシートの引き伸ばし力を強化し、Z方向への拡散を抑えた改良版。
#
# 必要なライブラリ: numpy, matplotlib, scipy
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors

# --- 物理定数・パラメータ ---
RM = 1.0              # 水星半径
M0 = 190.0            # 磁気モーメント強度
Z_OFF = 0.2           # ダイポール中心の北側ズレ
MAX_STEPS = 2000      # トレース最大ステップ数（尾を長く描くため増加）
DS = 0.04             # 刻み幅
DOMAIN_R = 15.0       # 計算ドメイン半径（X方向の後ろを長く）

def get_B_field(pos, Bx_imf, Bz_imf):
    """合成磁場ベクトル [Bx, By, Bz] の計算 (テイル強化版)"""
    x, y, z = pos[0], pos[1], pos[2]
    dz = z - Z_OFF
    
    r2 = x**2 + y**2 + dz**2
    r = np.sqrt(r2)
    
    # コア付近の特異点回避
    if r < 0.3:
        return np.array([0.0, 0.0, 0.0])
    
    r5 = r2**2 * r
    
    # 1. 水星の固有ダイポール (M0は南向き)
    Bx_dip = M0 * (-3 * x * dz) / r5
    By_dip = M0 * (-3 * y * dz) / r5
    Bz_dip = M0 * (x**2 + y**2 - 2 * dz**2) / r5
    
    # 2. 強力なテイル磁場 (カレントシート)
    # 夜側(x < 0)に向かって、北半球(z>0)は-X方向、南半球(z<0)は+X方向へ強く引っ張る
    B_tail_max = 60.0   # 尾を引く力（前回より強化）
    D = 0.8             # カレントシートの厚み
    
    # 昼夜の切り替え関数 (夜側 x<0 で効くように)
    tail_activation = 0.5 * (1.0 - np.tanh(x / 1.0))
    
    Bx_tail = -B_tail_max * tail_activation * np.tanh(dz / D)
    By_tail = 0.0
    # 発散をゼロにするためのZ成分の補正
    Bz_tail = -B_tail_max * (0.5 / np.cosh(x / 1.0)**2) * D * np.log(np.cosh(dz / D))

    # 3. 昼側のバリア (イメージダイポール)
    x_img = 2.0
    dx_img = x - x_img
    r_img2 = dx_img**2 + y**2 + dz**2
    r_img5 = r_img2**2 * np.sqrt(r_img2)
    M_img = M0 * 0.8  # 昼側のバリア強度
    
    Bx_img = M_img * (-3 * dx_img * dz) / r_img5
    By_img = M_img * (-3 * y * dz) / r_img5
    Bz_img = M_img * (dx_img**2 + y**2 - 2 * dz**2) / r_img5

    # 4. 全磁場の合成
    # IMFは、水星近傍(磁気圏内)ではシールドされる効果を簡易的に導入
    imf_penetration = 1.0 - np.exp(-r / 3.0)
    
    Bx = Bx_dip + Bx_tail + Bx_img + Bx_imf * imf_penetration
    By = By_dip + By_tail + By_img 
    Bz = Bz_dip + Bz_tail + Bz_img + Bz_imf * imf_penetration
    
    return np.array([Bx, By, Bz])

def rk2_step(pos, Bx_imf, Bz_imf, ds, forward=True):
    sign = 1.0 if forward else -1.0
    B1 = get_B_field(pos, Bx_imf, Bz_imf)
    B1_mag = np.linalg.norm(B1)
    if B1_mag < 1e-5: return pos, False 
    
    k1 = sign * (B1 / B1_mag)
    pos_mid = pos + 0.5 * ds * k1
    B2 = get_B_field(pos_mid, Bx_imf, Bz_imf)
    B2_mag = np.linalg.norm(B2)
    if B2_mag < 1e-5: return pos, False
        
    k2 = sign * (B2 / B2_mag)
    return pos + ds * k2, True

def trace_field_line(seed_pos, Bx_imf, Bz_imf):
    line = [seed_pos]
    hit_surface = 0
    
    # 前方向
    pos = np.copy(seed_pos)
    for _ in range(MAX_STEPS):
        pos, ok = rk2_step(pos, Bx_imf, Bz_imf, DS, forward=True)
        if not ok: break
        line.append(pos)
        r = np.linalg.norm(pos)
        if r <= RM:
            hit_surface += 1
            break
        if abs(pos[0]) > DOMAIN_R or abs(pos[2]) > 8.0: break # ドメイン外
            
    # 後方向
    pos = np.copy(seed_pos)
    line_back = []
    for _ in range(MAX_STEPS):
        pos, ok = rk2_step(pos, Bx_imf, Bz_imf, DS, forward=False)
        if not ok: break
        line_back.append(pos)
        r = np.linalg.norm(pos)
        if r <= RM:
            hit_surface += 1
            break
        if abs(pos[0]) > DOMAIN_R or abs(pos[2]) > 8.0: break
            
    full_line = np.array(line_back[::-1] + line)
    return full_line, hit_surface

# --- 描画セットアップ ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.05, bottom=0.25)

seeds = []
# 1. カスプ領域を狙い撃ちしたシード（赤いリコネクション線を確実に描くため）
theta_list = np.linspace(0, 2*np.pi, 8, endpoint=False)
for lat in [60, 68, 75, -60, -68, -75]:
    z = RM * np.sin(np.radians(lat))
    r_xy = RM * np.cos(np.radians(lat))
    for th in theta_list:
        seeds.append(np.array([r_xy*np.cos(th), r_xy*np.sin(th), z]))

# 2. 閉じた磁力線（青い線）用のシード
for lat in [10, 30, 45, -10, -30, -45]:
    z = RM * np.sin(np.radians(lat))
    r_xy = RM * np.cos(np.radians(lat))
    seeds.append(np.array([r_xy, 0.0, z]))

# 3. 太陽風（黄色い線）用のシード
for z in np.linspace(-6, 6, 10):
    seeds.append(np.array([5.0, 0.0, z]))

def draw_mercury(ax):
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = RM * np.outer(np.cos(u), np.sin(v))
    y = RM * np.outer(np.sin(u), np.sin(v))
    z = RM * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.5, edgecolor='none')
    ax.scatter([0], [0], [Z_OFF], color='red', s=50, label="Mag. Center")

def update_plot(val=None):
    Bx = slider_bx.val
    Bz = slider_bz.val
    
    ax.cla()
    draw_mercury(ax)
    
    # 尾（マイナスX方向）を非常に長く表示
    ax.set_xlim([-12, 6])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-6, 6])
    ax.set_xlabel('X (Toward Sun) [RM]')
    ax.set_ylabel('Y (Dusk) [RM]')
    ax.set_zlabel('Z (North) [RM]')
    ax.set_title(f'Mercury Magnetotail Topology\nIMF Bx: {Bx:.1f} nT, Bz: {Bz:.1f} nT', color='white')
    
    ax.set_facecolor('#050505')
    fig.patch.set_facecolor('#050505')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')

    for seed in seeds:
        line, top = trace_field_line(seed, Bx, Bz)
        if len(line) > 5:
            if top == 2:
                # 両端が地表：閉じた磁力線（水色）
                color, alpha, lw = '#4da6ff', 0.5, 1.0 
            elif top == 1:
                # 片方が地表、片方が宇宙：開いた磁力線/リコネクション（赤）
                color, alpha, lw = '#ff3333', 1.0, 2.0 
            else:
                # 両端が宇宙：太陽風（黄色）
                color, alpha, lw = '#ffaa00', 0.3, 1.0 
            ax.plot(line[:,0], line[:,1], line[:,2], color=color, alpha=alpha, linewidth=lw)
    
    # 真横（Y軸方向）から見ることで尾の長さを強調
    ax.view_init(elev=0, azim=-90)

axcolor = '#333333'
ax_bx = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
ax_bz = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor=axcolor)

slider_bx = Slider(ax_bx, 'IMF Bx (nT)', -40.0, 40.0, valinit=10.0, color='#ffcc00')
slider_bz = Slider(ax_bz, 'IMF Bz (nT)', -40.0, 40.0, valinit=-15.0, color='#ffcc00')

slider_bx.label.set_color('white')
slider_bz.label.set_color('white')
slider_bx.valtext.set_color('white')
slider_bz.valtext.set_color('white')

slider_bx.on_changed(update_plot)
slider_bz.on_changed(update_plot)

update_plot()
plt.show()