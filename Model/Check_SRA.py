import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

# =========================================================
# 1. 物理定数
# =========================================================
MASS_NA = 3.8175e-26
RM = 2.440e6
GM_MERCURY = 2.2032e13
GM_SUN = 1.32712440018e20
C = 299792458.0
AU_M = 1.496e11
MERCURY_A_AU = 0.387098
MERCURY_E = 0.205630
K_BOLTZMANN = 1.380649e-23
TEMP_BASE = 100.0
TEMP_AMP = 600.0
H_CONST = 6.62607015e-34

# 散乱断面積・フラックス計算用定数
const_sigma = (1.602e-19**2) / (4 * 9.109e-31 * 299792458.0 * 8.854e-12)
SIGMA0_1 = const_sigma * 0.320
SIGMA0_2 = const_sigma * 0.641
JL_CONST = 5.18e14

# =========================================================
# 2. 外部スペクトルデータの読み込み
# =========================================================
filename = 'SolarSpectrum_Na0.txt'
if not os.path.exists(filename):
    print(f"エラー: {filename} が見つかりません。シミュレーションと同じフォルダで実行してください。")
    sys.exit()

spec_np = np.loadtxt(filename, usecols=(0, 3))
spec_wl, spec_gamma = spec_np[:, 0], spec_np[:, 1]
if spec_wl[1] < spec_wl[0]:
    idx = np.argsort(spec_wl)
    spec_wl, spec_gamma = spec_wl[idx], spec_gamma[idx]

# =========================================================
# 3. 軌道パラメータ計算
# =========================================================
def get_orbit_params(taa_deg):
    taa_rad = np.deg2rad(taa_deg)
    p = MERCURY_A_AU * AU_M * (1.0 - MERCURY_E**2)
    r_m = p / (1.0 + MERCURY_E * np.cos(taa_rad))
    au = r_m / AU_M
    v_rad = np.sqrt(GM_SUN / p) * MERCURY_E * np.sin(taa_rad)
    return au, v_rad

# =========================================================
# 4. 放射圧(SRP)計算関数 (厳密計算)
# =========================================================
def calc_srp_exact(vx, v_rad, au):
    velocity_for_doppler = vx - v_rad
    w_na_d2 = 589.1582e-9 * (1.0 + velocity_for_doppler / C)
    w_na_d1 = 589.7558e-9 * (1.0 + velocity_for_doppler / C)
    
    # スペクトルデータからガンマ値を補間
    gamma2 = np.interp(w_na_d2 * 1e9, spec_wl, spec_gamma)
    gamma1 = np.interp(w_na_d1 * 1e9, spec_wl, spec_gamma)
    
    F_at_Merc = (JL_CONST * 1e13) / (au ** 2)
    term_d1 = (H_CONST / w_na_d1) * SIGMA0_1 * (F_at_Merc * gamma1 * w_na_d1 ** 2 / C)
    term_d2 = (H_CONST / w_na_d2) * SIGMA0_2 * (F_at_Merc * gamma2 * w_na_d2 ** 2 / C)
    
    b = (term_d1 + term_d2) / MASS_NA
    return b

# =========================================================
# 5. 粒子の軌道積分
# =========================================================
def simulate_particle(taa_deg, dt=1.0, max_time=5000.0):
    au, v_rad = get_orbit_params(taa_deg)
    
    scaling = np.sqrt(0.306 / au)
    t_surf = TEMP_BASE + TEMP_AMP * scaling 
    v_th = np.sqrt(2.0 * K_BOLTZMANN * t_surf / MASS_NA)
    
    x = RM
    v = v_th
    
    times = []
    srp_history = []
    
    t = 0.0
    while t < max_time:
        times.append(t)
        
        # 近似ではなく、厳密関数を使用
        a_srp = -calc_srp_exact(v, v_rad, au)
        srp_history.append(a_srp)
        
        a_grav = -GM_MERCURY / (x**2)
        
        v += (a_srp + a_grav) * dt
        x += v * dt
        t += dt
        
        if x <= RM or x >= RM * 6.0:
            break
            
    return np.array(times), np.array(srp_history)

# =========================================================
# 6. データ一括計算
# =========================================================
print("シミュレーションを実行中...")
taa_array = np.arange(0, 361, 5)
results = {}
total_srp_impulses = []
global_max_srp = 0.0  # 全TAAにおける最大の放射圧を記録

for taa in taa_array:
    t_hist, srp_hist = simulate_particle(taa)
    results[taa] = (t_hist, srp_hist)
    
    total_impulse = np.sum(np.abs(srp_hist)) * 1.0
    total_srp_impulses.append(total_impulse)
    
    # グラフ固定用の最大値を取得
    if len(srp_hist) > 0:
        current_max = np.max(np.abs(srp_hist))
        if current_max > global_max_srp:
            global_max_srp = current_max

print(f"計算完了。(最大放射圧: {global_max_srp:.2f} m/s^2)")

# =========================================================
# 7. Matplotlib グラフの構築
# =========================================================
fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25, hspace=0.45)

# --- 上段: 全体プロット ---
ax_total = fig.add_subplot(211)
ax_total.plot(taa_array, total_srp_impulses, '-o', color='royalblue')
current_taa_marker, = ax_total.plot([], [], 'ro', markersize=8) 
ax_total.set_xlabel('True Anomaly (TAA) [deg]')
ax_total.set_ylabel('Total SRP Impulse [m/s]')
ax_total.set_title('Total Solar Radiation Pressure received before falling')
ax_total.grid(True)
ax_total.set_xlim(0, 360)

# --- 下段: 詳細プロット ---
ax_detail = fig.add_subplot(212)
init_taa = 0
t_hist, srp_hist = results[init_taa]
line_detail, = ax_detail.plot(t_hist, np.abs(srp_hist), color='crimson')
ax_detail.set_xlabel('Time [s]')
ax_detail.set_ylabel('SRP Acceleration [m/s^2]')
ax_detail.set_title(f'SRP Acceleration over Time at TAA = {init_taa}°\n(Flight Time: {max(t_hist):.1f} s)')
ax_detail.grid(True)
ax_detail.set_xlim(0, max(t_hist)*1.1)
# 固定スケールを自動算出した最大値の1.1倍に設定
ax_detail.set_ylim(0, global_max_srp * 1.1)

# =========================================================
# 8. GIFアニメーションの出力
# =========================================================
print("GIFアニメーション (srp_animation.gif) を生成しています...")

def animate(frame_idx):
    taa = taa_array[frame_idx]
    t_h, srp_h = results[taa]
    
    line_detail.set_xdata(t_h)
    line_detail.set_ydata(np.abs(srp_h))
    
    # 飛行時間で横軸の幅は変動させる
    ax_detail.set_xlim(0, max(t_h)*1.1 if max(t_h) > 0 else 1.0)
    
    # 縦軸の幅は常に一定
    ax_detail.set_ylim(0, global_max_srp * 1.1)
    ax_detail.set_title(f'SRP Acceleration over Time at TAA = {taa}°\n(Flight Time: {max(t_h):.1f} s)')
    
    current_taa_marker.set_data([taa], [total_srp_impulses[frame_idx]])
    
    return line_detail, current_taa_marker

ani = FuncAnimation(fig, animate, frames=len(taa_array), interval=100, blit=True)
ani.save("srp_animation.gif", writer=PillowWriter(fps=10))
print("srp_animation.gif の保存が完了しました！ウィンドウを開きます。")

# =========================================================
# 9. インタラクティブ・スライダーの表示
# =========================================================
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(ax_slider, 'TAA [deg]', 0, 360, valinit=0, valstep=5)

def update_slider(val):
    taa = int(slider.val)
    idx = np.argmin(np.abs(taa_array - taa))
    animate(idx)
    fig.canvas.draw_idle()

slider.on_changed(update_slider)
current_taa_marker.set_data([0], [total_srp_impulses[0]])

plt.show()