# -*- coding: utf-8 -*-
"""
==============================================================================
概要
==============================================================================
シミュレーション結果の表面密度グリッド (.npy) を読み込みプロットします。

★修正点 (2025/11/24 v3)★
- ファイル検索ロジックを修正 (density_grid経由でTAAを特定)
- 太陽直下点の計算ロジックをシミュレーションコードと統一 (計算式ベースに変更)
- カラースケール固定機能などは維持
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
# BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202510"
BASE_OUTPUT_DIRECTORY = r"./SimulationResult_202511"

# 3. RUN名
RUN_NAME = f"DynamicGrid{N_LON}x{N_LAT}_16.0"  # 実行したフォルダ名に合わせてください
# RUN_NAME = f"SubCycle_{N_LON}x{N_LAT}_3.0"

# 4. プロット対象 (TAA指定 または 'latest')
FILE_TO_PLOT = 240

# 5. カラースケール設定
USE_LOG_SCALE = True

# 5b. カラースケール範囲の固定
FIX_COLOR_SCALE = True
COLOR_VMIN = 1.0e10  # 下限 [atoms/m^2]
COLOR_VMAX = 1.0e18  # 上限 [atoms/m^2]

# 6. 軌道データファイル名
ORBIT_FILE_PATH = 'orbit2025_v6.txt'

# 7. 太陽中心補正のON/OFF
ALIGN_SUN_TO_CENTER = True

# 物理定数
MERCURY_YEAR_SEC = 87.97 * 24 * 3600
ROTATION_PERIOD = 58.6462 * 86400


# ==============================================================================
# ヘルパー関数群
# ==============================================================================

def load_orbit_data(orbit_file_path):
    """軌道ファイルを読み込み、データ全体と基準時刻(TAA=0)を返します。"""
    try:
        orbit_data = np.loadtxt(orbit_file_path)
        # 0:TAA, 1:AU, 2:Time, ...
        taa_col = orbit_data[:, 0]
        time_col = orbit_data[:, 2]

        # 近日点(TAA=0)の時刻を探す
        idx_perihelion = np.argmin(np.abs(taa_col))
        t_start_run = time_col[idx_perihelion]

        print(f"軌道ファイル読み込み成功: 行数={len(orbit_data)}")
        print(f"基準時刻 (TAA=0): {t_start_run:.1f} s")
        return orbit_data, t_start_run
    except Exception as e:
        print(f"軌道ファイル読み込みエラー: {e}")
        sys.exit(1)


def calculate_subsolar_longitude(time_h, t_start_run, orbit_data):
    """
    指定時刻の太陽直下点経度を計算します（シミュレーションコードと同一ロジック）
    """
    relative_time_sec = float(time_h) * 3600.0
    current_t_sec = t_start_run + relative_time_sec

    # 軌道周期内での時刻
    cycle_sec = 87.969 * 86400  # ORBITAL_PERIOD
    dt_from_peri = (current_t_sec - t_start_run)
    time_in_cycle = dt_from_peri % cycle_sec

    # 軌道データからTAAを補間
    time_col_original = orbit_data[:, 2]
    # ファイル内の近日点時刻を取得して補正
    idx_peri = np.argmin(np.abs(orbit_data[:, 0]))
    t_peri_in_file = time_col_original[idx_peri]
    t_lookup = t_peri_in_file + time_in_cycle

    taa_deg = np.interp(t_lookup, time_col_original, orbit_data[:, 0])
    taa_rad = np.deg2rad(taa_deg)

    # 自転角の計算
    omega_rot = 2 * np.pi / ROTATION_PERIOD
    rotation_angle = omega_rot * dt_from_peri

    # 太陽直下点経度 (Subsolar Longitude)
    subsolar_lon_rad = taa_rad - rotation_angle
    subsolar_lon_rad = (subsolar_lon_rad + np.pi) % (2 * np.pi) - np.pi

    return subsolar_lon_rad


def find_target_file(target_dir, preference):
    """
    density_grid (TAA付き) を検索してIDを特定し、surface_density ファイルを返す
    """
    # まず density_grid_*.npy を探す (TAA情報が含まれているため)
    search_path_grid = os.path.join(target_dir, "density_grid_*.npy")
    grid_files = glob.glob(search_path_grid)

    if not grid_files:
        print("エラー: density_grid ファイルが見つかりません。")
        return None

    # (TAA, TimeID, FullPath) のリストを作成
    candidates = []
    for f in grid_files:
        # density_grid_tXXXXX_taaYYY.npy
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', os.path.basename(f))
        if match:
            time_id = int(match.group(1))  # tXXXXX
            taa_val = int(match.group(2))  # taaYYY
            candidates.append((taa_val, time_id, f))

    if not candidates:
        return None

    # TAAでソート
    candidates.sort(key=lambda x: x[0])

    selected_time_id = None
    target_taa_disp = "Latest"

    if isinstance(preference, (int, float)):
        # 指定されたTAAに近いものを探す
        target_taa = int(preference)
        # 完全一致を探す
        matches = [c for c in candidates if c[0] == target_taa]
        if matches:
            # 複数ある場合は最後（最新）のもの
            selected_time_id = matches[-1][1]
            target_taa_disp = matches[-1][0]
        else:
            # 近似検索する場合（オプション）
            print(f"TAA={target_taa} の完全一致なし。最新を表示します。")
            selected_time_id = candidates[-1][1]
            target_taa_disp = candidates[-1][0]
    else:
        # 'latest' の場合
        selected_time_id = candidates[-1][1]
        target_taa_disp = candidates[-1][0]

    # 表面密度ファイルのパスを構築
    # surface_density_tXXXXX.npy
    surf_filename = f"surface_density_t{selected_time_id:05d}.npy"
    surf_filepath = os.path.join(target_dir, surf_filename)

    if not os.path.exists(surf_filepath):
        print(f"エラー: 対応する表面ファイルが見つかりません -> {surf_filename}")
        return None

    print(f"Plotting Target: TAA={target_taa_disp} (TimeID={selected_time_id})")
    return surf_filepath, selected_time_id


# ==============================================================================
# プロット関数
# ==============================================================================

def plot_surface_grid(filepath, time_h, n_lon, n_lat, use_log, t_start_run, orbit_data,
                      align_sun=True,
                      fix_scale=False, vmin_fixed=None, vmax_fixed=None):
    try:
        data_fixed = np.load(filepath)
    except Exception as e:
        print(e)
        return

    if data_fixed.shape != (n_lon, n_lat):
        print(f"形状不一致: Data{data_fixed.shape} != Config({n_lon}, {n_lat})")
        return

    # --- 軌道計算 (修正版関数を使用) ---
    subsolar_lon_rad_fixed = calculate_subsolar_longitude(time_h, t_start_run, orbit_data)
    subsolar_lon_deg_fixed = np.rad2deg(subsolar_lon_rad_fixed)

    # --- データの回転処理 ---
    if align_sun:
        dlon_deg = 360.0 / n_lon
        # 太陽がグリッド上のどこにあるか (index)
        sun_index_float = (subsolar_lon_deg_fixed + 180.0) / dlon_deg
        sun_index = int(np.round(sun_index_float)) % n_lon

        # 画像の中心 (index)
        center_index = n_lon // 2

        # 太陽(sun_index) を 中心(center_index) に持ってくるためのシフト量
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

    # スケール設定
    if fix_scale and (vmin_fixed is not None) and (vmax_fixed is not None):
        final_vmin = vmin_fixed
        final_vmax = vmax_fixed
    else:
        valid_min_auto = 1e8
        if np.any(data_to_plot_T > 0):
            valid_min_auto = np.min(data_to_plot_T[data_to_plot_T > 0])
        final_vmin = valid_min_auto
        final_vmax = np.max(data_to_plot_T)

    if use_log:
        if final_vmin <= 0: final_vmin = 1e-10
        norm = LogNorm(vmin=final_vmin, vmax=final_vmax)
    else:
        norm = Normalize(vmin=final_vmin, vmax=final_vmax)

    # --- 描画 ---
    lon_edges_deg = np.linspace(-180, 180, n_lon + 1)
    lat_edges_deg = np.linspace(-90, 90, n_lat + 1)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

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
    if not os.path.exists(ORBIT_FILE_PATH):
        print(f"エラー: 軌道ファイルがありません {ORBIT_FILE_PATH}")
        sys.exit(1)

    # 1. 軌道データを読み込む
    orbit_data_full, t_start = load_orbit_data(ORBIT_FILE_PATH)

    # 2. ファイルを探す
    full_output_dir = os.path.join(BASE_OUTPUT_DIRECTORY, RUN_NAME)

    if not os.path.exists(full_output_dir):
        print(f"エラー: 結果フォルダが見つかりません {full_output_dir}")
        sys.exit(1)

    res = find_target_file(full_output_dir, FILE_TO_PLOT)

    if res:
        fpath, th = res
        # 3. プロット実行
        plot_surface_grid(
            fpath, th, N_LON, N_LAT, USE_LOG_SCALE,
            t_start, orbit_data_full,
            align_sun=ALIGN_SUN_TO_CENTER,
            fix_scale=FIX_COLOR_SCALE,
            vmin_fixed=COLOR_VMIN,
            vmax_fixed=COLOR_VMAX
        )
    else:
        print("プロット可能なファイルが見つかりませんでした。")