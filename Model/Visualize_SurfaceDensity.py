# -*- coding: utf-8 -*-
"""
==============================================================================
概要
==============================================================================
シミュレーション結果の表面密度グリッド (.npy) を読み込みプロットします。

★修正点 (2025/11/24 v2)★
- カラースケール（カラーバー）の最大値・最小値を固定する機能を追加。
- 太陽直下点を中央にする補正機能（前回追加分）も維持。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import glob
import os
import re
import sys

# ==============================================================================
# 【設定】
# ==============================================================================

# 1. グリッド解像度
N_LON, N_LAT = 72, 36

# 2. 出力ディレクトリ
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202510"
#BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202511"

# 3. RUN名
RUN_NAME = f"DynamicGrid{N_LON}x{N_LAT}_10.0"
#RUN_NAME = f"SubCycle_{N_LON}x{N_LAT}_3.0"

# 4. プロット対象 (TAA指定 または 'latest')
FILE_TO_PLOT = 100

# 5. カラースケール設定
USE_LOG_SCALE = True

# ★★★ 5b. カラースケール範囲の固定 (新規追加) ★★★
# Trueにすると、以下の COLOR_VMIN, COLOR_VMAX で色を固定します。
# Falseにすると、そのデータの最小値・最大値に合わせて自動調整されます。
FIX_COLOR_SCALE = True

# 固定する場合の値 (USE_LOG_SCALE=True の場合は、0より大きい値を指定してください)
COLOR_VMIN = 1.0e10  # 下限 [atoms/m^2]
COLOR_VMAX = 1.0e18  # 上限 [atoms/m^2]

# 6. 軌道データファイル名
ORBIT_FILE_PATH = 'orbit2025_v6.txt'

# 7. 太陽中心補正のON/OFF
# True: 太陽直下点が画像の中心(0度)に来るようにグリッドを回転させる (MSO風)
ALIGN_SUN_TO_CENTER = True

# 8. 手動回転補正 (予備)
MANUAL_OFFSET_DEG = 0.0

# 水星の1公転周期 [s]
MERCURY_YEAR_SEC = 87.97 * 24 * 3600


# ==============================================================================
# ヘルパー関数群
# ==============================================================================

def load_orbit_data(orbit_file_path):
    """軌道ファイルを読み込み、データ全体と基準時刻(TAA=0)を返します。"""
    try:
        orbit_data = np.loadtxt(orbit_file_path)
        taa_col = orbit_data[:, 0]
        time_col = orbit_data[:, 2]
        idx_perihelion = np.argmin(np.abs(taa_col))
        t_start_run = time_col[idx_perihelion]
        print(f"軌道ファイル読み込み成功: 行数={len(orbit_data)}")
        print(f"基準時刻 (TAA=0): {t_start_run:.1f} s")
        return orbit_data, t_start_run
    except Exception as e:
        print(f"軌道ファイル読み込みエラー: {e}")
        sys.exit(1)


def get_subsolar_longitude_from_file(time_h, t_start_run, orbit_data):
    """指定時刻の太陽直下点経度を軌道ファイルから取得"""
    relative_time_sec = float(time_h) * 3600.0
    current_t_sec = t_start_run + relative_time_sec

    # ★重要: 水星の太陽直下点は「2公転(176日)」で1周します。
    current_time_in_orbit = current_t_sec % MERCURY_YEAR_SEC

    orbit_time_col = orbit_data[:, 2]
    orbit_subsolar_col = orbit_data[:, 5]  # 5列目(太陽直下点経度)

    subsolar_lon_deg = np.interp(current_time_in_orbit, orbit_time_col, orbit_subsolar_col)
    subsolar_lon_rad = np.deg2rad(subsolar_lon_deg)
    subsolar_lon_rad = (subsolar_lon_rad + np.pi) % (2 * np.pi) - np.pi
    return subsolar_lon_rad


def find_target_file(target_dir, preference):
    """指定された条件に合うファイルを探す"""
    search_path = os.path.join(target_dir, "surface_density_*.npy")
    files = glob.glob(search_path)
    if not files: return None

    file_info_list = []
    for f in files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            file_info_list.append((int(match.group(1)), int(match.group(2)), f))

    if not file_info_list: return None
    file_info_list.sort(key=lambda x: x[0])

    if isinstance(preference, (int, float)):
        target_taa = int(preference)
        candidates = [x for x in file_info_list if x[1] == target_taa]
        if candidates: return candidates[-1][2], candidates[-1][0]
        print(f"TAA={target_taa} が見つかりません。最新を表示します。")

    return file_info_list[-1][2], file_info_list[-1][0]


# ==============================================================================
# プロット関数 (修正版)
# ==============================================================================

def plot_surface_grid(filepath, time_h, n_lon, n_lat, use_log, t_start_run, orbit_data,
                      align_sun=True,
                      fix_scale=False, vmin_fixed=None, vmax_fixed=None):
    try:
        data_fixed = np.load(filepath)
    except Exception as e:
        print(e);
        return

    if data_fixed.shape != (n_lon, n_lat):
        print(f"形状不一致: Data{data_fixed.shape} != Config({n_lon}, {n_lat})");
        return

    # --- 軌道計算 ---
    subsolar_lon_rad_fixed = get_subsolar_longitude_from_file(time_h, t_start_run, orbit_data)
    subsolar_lon_deg_fixed = np.rad2deg(subsolar_lon_rad_fixed)

    # --- データの回転処理 ---
    if align_sun:
        dlon_deg = 360.0 / n_lon
        sun_index_float = (subsolar_lon_deg_fixed + 180.0) / dlon_deg
        sun_index = int(np.round(sun_index_float)) % n_lon
        center_index = n_lon // 2
        shift_amount = center_index - sun_index
        data_to_plot = np.roll(data_fixed, shift=shift_amount, axis=0)

        plot_title_mode = "(Geometric Aligned: Sun Center)"
        xlabel_text = 'Longitude (Sun-Fixed / MSO) [degrees]'
        print(f"--- Alignment: ON ---")
        print(f"Sun Lon: {subsolar_lon_deg_fixed:.2f} deg -> Shift Grid by {shift_amount}")
    else:
        data_to_plot = data_fixed
        plot_title_mode = "(Raw Data: Planet Fixed)"
        xlabel_text = 'Longitude (Planet-Fixed) [degrees]'
        print(f"--- Alignment: OFF ---")

    # --- プロット準備 ---
    data_to_plot_T = data_to_plot.T

    # 自動スケール用の最小値・最大値計算
    valid_min_auto = 1e8
    if np.any(data_to_plot_T > 0):
        valid_min_auto = np.min(data_to_plot_T[data_to_plot_T > 0])
    max_auto = np.max(data_to_plot_T)

    # 最終的な vmin, vmax の決定
    if fix_scale and (vmin_fixed is not None) and (vmax_fixed is not None):
        # 固定モード
        final_vmin = vmin_fixed
        final_vmax = vmax_fixed
    else:
        # 自動モード
        final_vmin = valid_min_auto
        final_vmax = max_auto

    # ノルム(色割り当て)の設定
    if use_log:
        # ログスケールで vmin <= 0 だとエラーになるため保護
        if final_vmin <= 0:
            final_vmin = 1e-10  # 極小値を入れる
        norm = LogNorm(vmin=final_vmin, vmax=final_vmax)
    else:
        norm = Normalize(vmin=final_vmin, vmax=final_vmax)

    # --- 描画 ---
    lon_edges_deg = np.linspace(-180, 180, n_lon + 1)
    lat_edges_deg = np.linspace(-90, 90, n_lat + 1)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # pcolormeshでは norm を指定すれば vmin/vmax 引数は不要
    mesh = ax.pcolormesh(lon_edges_deg, lat_edges_deg, data_to_plot_T,
                         shading='flat', cmap='inferno', norm=norm)

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('Surface Density [atoms/m^2]')

    ax.set_xlabel(xlabel_text)
    ax.set_ylabel('Latitude [degrees]')
    ax.set_title(f'Surface Density {plot_title_mode}\nTime: {time_h}h (Sun Lon: {subsolar_lon_deg_fixed:.1f}deg)')

    ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_yticks(np.arange(-90, 91, 30))

    # 補助線
    if align_sun:
        ax.axvline(0, color='white', linestyle='--', linewidth=1.5, label='Noon (Sun)')
    else:
        ax.axvline(subsolar_lon_deg_fixed, color='white', linestyle='--', linewidth=1.5, label='Sun Position')

    ax.axvline(-180, color='cyan', linestyle=':', linewidth=1.5)
    ax.axvline(180, color='cyan', linestyle=':', linewidth=1.5)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# メイン実行部
# ==============================================================================

if __name__ == "__main__":
    # 1. 軌道データを読み込む
    orbit_data_full, t_start = load_orbit_data(ORBIT_FILE_PATH)

    # 2. ファイルを探す
    full_output_dir = os.path.join(BASE_OUTPUT_DIRECTORY, RUN_NAME)
    res = find_target_file(full_output_dir, FILE_TO_PLOT)

    if res:
        fpath, th = res
        # 3. プロット実行
        # 設定変数を引数として渡します
        plot_surface_grid(
            fpath, th, N_LON, N_LAT, USE_LOG_SCALE,
            t_start, orbit_data_full,
            align_sun=ALIGN_SUN_TO_CENTER,
            fix_scale=FIX_COLOR_SCALE,
            vmin_fixed=COLOR_VMIN,
            vmax_fixed=COLOR_VMAX
        )
    else:
        print("ファイルが見つかりませんでした。終了します。")