# -*- coding: utf-8 -*-
"""
==============================================================================
水星ナトリウム外気圏: 4つの粒子生成過程 (PSD/TD/SWS/MMV) の
表面位置に対する規格化強度を3D球面上に可視化するスクリプト
==============================================================================

元コード (mkNaColumnDensity9_9.py) の中で使われている各生成過程の
物理式 (update_surface_maps_numba 内の r_psd, r_td, r_sws および
calculate_mmv_flux) をそのまま踏襲し、瞬間的な表面フラックス分布として
再計算・可視化する。

- PSD (光脱離)                 : 太陽天頂角の余弦(照度)に比例 → 昼側で最大、夜側でゼロ
- TD  (熱脱離)                  : 表面温度(昼側で高温)に対する指数関数
                                   → 束縛エネルギーのガウス分布(V_weights)で加重平均した代表値
- SWS (太陽風スパッタリング)     : 極域の限られた経度・緯度帯(SWS_PARAMSのマスク領域)でのみ一定値
- MMV (微隕石衝突による蒸発)     : 表面位置に依存しない(球全体で一様) → 規格化後は全面 1.0

画面下部のボタンで表示するプロセスを切り替えられる。

実行環境: matplotlibのインタラクティブGUIバックエンド(TkAgg, QtAggなど)が必要です。
    $ python plot_mercury_source_processes.py
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import japanize_matplotlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection registration)

# ==============================================================================
# 元コードから引用した物理定数・設定 (mkNaColumnDensity9_9.py の値をそのまま使用)
# ==============================================================================
RM = 2.440e6                       # 水星半径 [m]
K_BOLTZMANN = 1.380649e-23         # ボルツマン定数 [J/K]
EV_TO_JOULE = 1.602e-19            # eV -> J
A_MERCURY_AU = 0.387098            # 水星軌道長半径 [AU]
E_MERCURY = 0.205630               # 水星軌道離心率

# 温度モデル (Leblanc et al.)
TEMP_BASE = 100.0
TEMP_AMP = 600.0
TEMP_NIGHT = 100.0

# --- PSDパラメータ ---
F_UV_1AU = 1.5e14 * (100 ** 2)       # 1AUでのUVフラックス [photons/m^2/s]
Q_PSD_BASE = 2.0e-20 / (100 ** 2)    # --q_psd_base=2.0 (デフォルト値)相当

# --- TDパラメータ (--u_model=gaussian_random のデフォルト値) ---
U_MU, U_SIGMA = 1.85, 0.25
N_U_BINS, U_MIN, U_MAX = 10, 1.4, 2.7

# --- SWSパラメータ ---
SWS_PARAMS = {
    'FLUX_1AU': 10.0 * 100 ** 3 * 400e3 * 4,
    'YIELD': 0.06,
    'REF_DENS': 7.5e14 * 100 ** 2,
    'LON_RANGE': np.deg2rad([-40, 40]),
    'LAT_N_RANGE': np.deg2rad([20, 80]),
    'LAT_S_RANGE': np.deg2rad([-80, -20]),
}

# ==============================================================================
# 表示する軌道位置(真近点角 TAA)。太陽距離(AU)と温度振幅がここで決まる。
# 太陽直下点の経度は SUB_LON=0 に固定(位置分布の"形"はTAAを変えても同じ)。
# ==============================================================================
TAA_DEG = 0.0   # 0度 = 近日点
SUB_LON = 0.0


def calculate_au_at_taa(taa_deg):
    rad = np.deg2rad(taa_deg)
    return A_MERCURY_AU * (1 - E_MERCURY ** 2) / (1 + E_MERCURY * np.cos(rad))


def calculate_mmv_flux(au):
    TOTAL_FLUX_AT_PERI = 5e23
    PERIHELION_AU = 0.307
    area = 4 * np.pi * (RM ** 2)
    avg_flux_peri = TOTAL_FLUX_AT_PERI / area
    c = avg_flux_peri * (PERIHELION_AU ** 1.9)
    return c * (au ** (-1.9))


def setup_binding_energy_bins():
    u_bins = np.linspace(U_MIN, U_MAX, N_U_BINS)
    v_weights = np.exp(-0.5 * ((u_bins - U_MU) / U_SIGMA) ** 2)
    v_weights /= np.sum(v_weights)
    return u_bins, v_weights


U_BINS, V_WEIGHTS = setup_binding_energy_bins()

# ==============================================================================
# 経度・緯度グリッド (球面を滑らかに描くため元設定 N_LON=72,N_LAT=36 より高解像度)
# ==============================================================================
N_LON, N_LAT = 180, 90
lon_edges = np.linspace(-np.pi, np.pi, N_LON + 1)
lat_edges = np.linspace(-np.pi / 2, np.pi / 2, N_LAT + 1)
lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
LON, LAT = np.meshgrid(lon_centers, lat_centers, indexing='ij')  # shape (N_LON, N_LAT)


def compute_intensities(taa_deg):
    """元コード update_surface_maps_numba 内の各レート式を再現して表面分布を計算する"""
    au = calculate_au_at_taa(taa_deg)
    scaling = np.sqrt(0.306 / au)

    dlon = lon_centers[1] - lon_centers[0]
    sin_half_width = np.sin(dlon / 2.0)

    cos_z = np.cos(LAT) * np.cos(LON - SUB_LON)
    eff_cos = np.clip(cos_z, 0.0, None)
    # セル端でのソフトな昼夜境界(元コードのillum計算と同じ)
    illum = np.clip((cos_z + sin_half_width) / (2.0 * sin_half_width), 0.0, 1.0)

    t_day = TEMP_BASE + TEMP_AMP * (eff_cos ** 0.25) * scaling

    # --- PSD (光脱離): 照度 x 天頂角余弦 に比例 ---
    f_uv = F_UV_1AU / (au ** 2)
    flux_psd = f_uv * Q_PSD_BASE * eff_cos * illum

    # --- TD (熱脱離): 束縛エネルギー分布(V_WEIGHTS)で加重平均した代表フラックス ---
    flux_td = np.zeros_like(LON)
    for u_ev, w in zip(U_BINS, V_WEIGHTS):
        u_j = u_ev * EV_TO_JOULE
        rate_day = np.where(t_day >= 10.0,
                             1e13 * np.exp(-u_j / (K_BOLTZMANN * np.maximum(t_day, 10.0))),
                             0.0)
        rate_night = 1e13 * np.exp(-u_j / (K_BOLTZMANN * TEMP_NIGHT))
        flux_td += w * (rate_day * illum + rate_night * (1.0 - illum))

    # --- SWS (太陽風スパッタリング): 極域カスプ相当のマスク領域のみ一定値 ---
    sw_flux = SWS_PARAMS['FLUX_1AU'] / (au ** 2)
    lon_sun = (LON - SUB_LON + np.pi) % (2 * np.pi) - np.pi
    mask_lon = (SWS_PARAMS['LON_RANGE'][0] <= lon_sun) & (lon_sun <= SWS_PARAMS['LON_RANGE'][1])
    mask_lat = ((SWS_PARAMS['LAT_N_RANGE'][0] <= LAT) & (LAT <= SWS_PARAMS['LAT_N_RANGE'][1])) | \
               ((SWS_PARAMS['LAT_S_RANGE'][0] <= LAT) & (LAT <= SWS_PARAMS['LAT_S_RANGE'][1]))
    flux_sws = np.where(mask_lon & mask_lat,
                         sw_flux * SWS_PARAMS['YIELD'] / SWS_PARAMS['REF_DENS'],
                         0.0)

    # --- MMV (微隕石衝突蒸発): 位置に依存しない一様フラックス ---
    flux_mmv = np.full_like(LON, calculate_mmv_flux(au))

    return {
        'PSD': ('PSD (光脱離)', flux_psd),
        'TD': ('TD (熱脱離)', flux_td),
        'SWS': ('SWS (太陽風スパッタリング)', flux_sws),
        'MMV': ('MMV (微隕石衝突蒸発)', flux_mmv),
    }


def normalize(field):
    m = np.max(field)
    return field / m if m > 0 else np.zeros_like(field)


raw = compute_intensities(TAA_DEG)
intensities = {key: (label, normalize(field)) for key, (label, field) in raw.items()}
process_keys = ['PSD', 'TD', 'SWS', 'MMV']

# ==============================================================================
# 水星表面の3D座標 (半径1に規格化、+X方向が太陽直下点=SUB_LON)
# ==============================================================================
X = np.cos(LAT) * np.cos(LON)
Y = np.cos(LAT) * np.sin(LON)
Z = np.sin(LAT)

# ==============================================================================
# 描画
# ==============================================================================
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.18)

cmap = plt.cm.inferno
mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_clim(0.0, 1.0)
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
cbar.set_label('規格化された生成率')


def draw(key):
    label, values = intensities[key]
    ax.clear()
    facecolors = cmap(values)
    ax.plot_surface(X, Y, Z, facecolors=facecolors, rstride=1, cstride=1,
                     linewidth=0, antialiased=False, shade=False)
    ax.scatter([1], [0], [0], color='yellow', s=90, edgecolor='k', zorder=10,
               label='太陽直下点')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X (太陽方向+)')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (北極+)')
    ax.set_title(f"{label}\n表面位置に対する規格化された生成率 (TAA={TAA_DEG:.0f}°)")
    ax.legend(loc='upper left')
    ax.view_init(elev=20, azim=-60)
    fig.canvas.draw_idle()


draw(process_keys[0])

# --- プロセス切り替えボタン ---
buttons = []
n = len(process_keys)
btn_width = 0.18
start_x = 0.5 - (n * btn_width) / 2

for i, key in enumerate(process_keys):
    bax = plt.axes([start_x + i * btn_width, 0.03, btn_width - 0.015, 0.06])
    b = Button(bax, key)

    def make_callback(k):
        return lambda event: draw(k)

    b.on_clicked(make_callback(key))
    buttons.append(b)

plt.show()