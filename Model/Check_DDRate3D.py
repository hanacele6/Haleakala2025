# -*- coding: utf-8 -*-
"""
水星表面 Na拡散フラックス 3Dプロット (Logスケール版)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm  # <--- 追加: 対数スケール用

# ==============================================================================
# 1. 物理定数・設定
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'RM': 2.440e6,  # [m] 水星半径
    'KB_EV': 8.617e-5,  # [eV/K] ボルツマン定数
    # 軌道計算用
    'MERCURY_SEMI_MAJOR_AXIS_AU': 0.387098,
    'MERCURY_ECCENTRICITY': 0.205630,
}

# --- 拡散モデルパラメータ ---
DIFF_REF_FLUX = 5.0e7 * (100.0 ** 2)  # [atoms/m^2/s] Reference
DIFF_REF_TEMP = 700.0  # [K]
DIFF_E_A_EV = 0.4  # [eV] 活性化エネルギー

# 頻度因子 A の計算
DIFF_PRE_FACTOR = DIFF_REF_FLUX / np.exp(-DIFF_E_A_EV / (PHYSICAL_CONSTANTS['KB_EV'] * DIFF_REF_TEMP))

# --- Clamp (Peak-Cut) 設定 ---
TAA_CLAMP_START = None
TAA_CLAMP_END = None

# --- 温度モデル設定 (Leblanc et al.) ---
TEMP_BASE = 100.0
TEMP_AMP = 600.0
TEMP_NIGHT = 100.0


# ==============================================================================
# 2. 計算関数群
# ==============================================================================

def calculate_au_from_taa(taa_deg):
    """TAA(度)から水星の太陽距離(AU)を計算"""
    a = PHYSICAL_CONSTANTS['MERCURY_SEMI_MAJOR_AXIS_AU']
    e = PHYSICAL_CONSTANTS['MERCURY_ECCENTRICITY']
    rad = np.deg2rad(taa_deg)
    r = a * (1 - e ** 2) / (1 + e * np.cos(rad))
    return r


def get_effective_au(taa_deg):
    """
    TAAに基づき、拡散計算に使用する「実効的な距離(AU)」を決定する
    """
    au_actual = calculate_au_from_taa(taa_deg)

    if TAA_CLAMP_START is None or TAA_CLAMP_END is None:
        return au_actual, False

    if TAA_CLAMP_START <= taa_deg <= TAA_CLAMP_END:
        return au_actual, False
    else:
        au_cutoff = calculate_au_from_taa(TAA_CLAMP_START)
        return au_cutoff, True


def calculate_surface_temperature_field(lat_grid, lon_grid, au_distance):
    """球面グリッド上の温度分布を計算"""
    cos_theta = np.cos(lat_grid) * np.cos(lon_grid)
    scaling = np.sqrt(0.306 / au_distance)

    T_field = np.zeros_like(lat_grid)
    day_mask = cos_theta > 0

    T_field[day_mask] = TEMP_BASE + TEMP_AMP * (cos_theta[day_mask] ** 0.25) * scaling
    T_field[~day_mask] = TEMP_NIGHT

    return T_field


def calculate_flux_field(temp_field):
    """温度フィールドから拡散フラックスフィールドを計算"""
    flux_field = np.zeros_like(temp_field)
    valid_mask = temp_field > 105.0  # 低温カットオフ

    kt = PHYSICAL_CONSTANTS['KB_EV'] * temp_field[valid_mask]
    flux_field[valid_mask] = DIFF_PRE_FACTOR * np.exp(-DIFF_E_A_EV / kt)

    return flux_field


# ==============================================================================
# 3. メイン処理・プロット
# ==============================================================================

if __name__ == '__main__':
    # --- ユーザー設定 ---
    TARGET_TAA = 180.0  # 近日点(0), 遠日点(180)

    # プロットの最大値・最小値設定
    # 【重要】対数スケールの場合、VMINは 0 より大きい必要があります (例: 1.0e5)
    # 値が小さい領域も色で見たい場合は、この値を下げてください（例: 1.0e4）
    PLOT_VMIN = 1.0e5
    PLOT_VMAX = None  # Noneの場合はデータの最大値を使用

    # ------------------

    # --- 1. 計算 ---
    au_eff, is_clamped = get_effective_au(TARGET_TAA)
    au_real = calculate_au_from_taa(TARGET_TAA)

    print(f"--- Calculation Conditions ---")
    print(f"Target TAA: {TARGET_TAA} deg")
    print(f"Real AU:    {au_real:.4f} AU")

    # グリッド生成
    lon = np.linspace(-np.pi, np.pi, 200)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 100)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 分布計算
    T_grid = calculate_surface_temperature_field(lat_grid, lon_grid, au_eff)
    Flux_grid_m2 = calculate_flux_field(T_grid)
    Flux_grid_cm2 = Flux_grid_m2 / 1.0e4  # [atoms/cm^2/s]

    # 3D座標変換
    X = PHYSICAL_CONSTANTS['RM'] * np.cos(lat_grid) * np.cos(lon_grid)
    Y = PHYSICAL_CONSTANTS['RM'] * np.cos(lat_grid) * np.sin(lon_grid)
    Z = PHYSICAL_CONSTANTS['RM'] * np.sin(lat_grid)

    # --- 2. プロット設定 ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    cmap_style = plt.cm.inferno

    # カラーバー範囲の決定
    # LogScaleのため、vminは正の値である必要があります。
    # データに0が含まれていても、ここで指定した vmin 以下の値は
    # カラーマップの最低色（黒など）にマッピングされます。
    vmin_val = PLOT_VMIN if PLOT_VMIN is not None else 1.0e5
    vmax_val = PLOT_VMAX if PLOT_VMAX is not None else np.max(Flux_grid_cm2)

    # 正規化オブジェクトを LogNorm に変更
    norm = LogNorm(vmin=vmin_val, vmax=vmax_val)

    # プロット
    surf = ax.plot_surface(
        X, Y, Z,
        rstride=2, cstride=2,
        facecolors=cmap_style(norm(Flux_grid_cm2)),  # LogNormを適用
        linewidth=0,
        antialiased=False,
        shade=False
    )

    # --- カラーバー ---
    mappable = plt.cm.ScalarMappable(cmap=cmap_style, norm=norm)
    mappable.set_array(Flux_grid_cm2)

    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    # ラベルに (Log Scale) を追加しておくと親切です
    cbar.set_label(r'Na Diffusion Flux [$atoms/cm^2/s$] (Log Scale)', rotation=270, labelpad=20)

    # --- グラフ装飾 ---
    ax.set_title(f'Na Diffusion Flux Distribution (Log Scale)\nTAA={TARGET_TAA}° (AU={au_eff:.3f})')
    ax.set_xlabel('X [m] (Subsolar)')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    max_range = PHYSICAL_CONSTANTS['RM']
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30., azim=0)

    # 結果表示
    max_flux = np.max(Flux_grid_cm2)
    print(f"--- Results ---")
    print(f"Max Flux (Calc):  {max_flux:.2e} atoms/cm^2/s")
    print(f"Plot Range (Log): {vmin_val:.1e} - {vmax_val:.1e}")

    plt.show()