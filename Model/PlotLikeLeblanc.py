# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob

# ==============================================================================
# 設定
# ==============================================================================
#RESULT_DIR = r"./SimulationResult_202511/DynamicGrid72x36_18.0"
RESULT_DIR = r"./SimulationResult_202512/DynamicGrid72x36_NoEq"
#RESULT_DIR = r"./SimulationResult_202511/SubCycle_72x36_2.0"
#RESULT_DIR = r"./SimulationResult_202512/SmartSample_72x36_1.0"

RM = 2.440e6  # 水星半径 [m]
PI = np.pi

# グリッド設定
N_LON_FIXED = 72
N_LAT = 36
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# プロット設定
PLOT_SURFACE = True

# ★追加設定: 補正のON/OFF
APPLY_CORRECTION = False

# ★補正係数 (宿題3の考察に基づく)
CORRECTION_FACTOR = 1.0 / 0.50


# ==============================================================================
# ヘルパー関数
# ==============================================================================
def calculate_cell_areas():
    """表面グリッドの各セルの面積 [m^2] を計算する"""
    lon_edges = np.linspace(-PI, PI, N_LON_FIXED + 1)
    lat_edges = np.linspace(-PI / 2, PI / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]
    lat_term = np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    cell_areas_1d = (RM ** 2) * dlon * lat_term
    return np.tile(cell_areas_1d, (N_LON_FIXED, 1))


def calculate_voxel_volume():
    """空間グリッドの1ボクセルあたりの体積 [m^3] を計算する"""
    gmin = -GRID_MAX_RM * RM
    gmax = GRID_MAX_RM * RM
    csize = (gmax - gmin) / GRID_RESOLUTION
    return csize ** 3


def parse_filename_info(filename):
    """
    ファイル名から時刻IDとTAAを抽出する
    例: density_grid_t00123_taa045.npy
    戻り値: (time_str, taa_val) -> ("t00123", 45)
    """
    t_match = re.search(r'(t\d{5})', filename)
    taa_match = re.search(r'taa(\d+)', filename)

    if t_match and taa_match:
        return t_match.group(1), int(taa_match.group(1))
    return None, None


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    if not os.path.exists(RESULT_DIR):
        print(f"エラー: ディレクトリ '{RESULT_DIR}' が見つかりません。")
        return

    density_files = glob.glob(os.path.join(RESULT_DIR, "density_grid_*.npy"))

    if not density_files:
        print("データファイル (density_grid_*.npy) が見つかりません。")
        return

    data_list = []
    cell_areas = calculate_cell_areas() if PLOT_SURFACE else None
    voxel_volume = calculate_voxel_volume()

    print(f"データを読み込み中... ({len(density_files)} files found)")

    for d_file in sorted(density_files):
        time_id, taa = parse_filename_info(os.path.basename(d_file))

        if time_id is None:
            continue

        # 外気圏データの読み込み
        try:
            e_data = np.load(d_file)
            total_exosphere = np.sum(e_data) * voxel_volume
        except Exception as e:
            print(f"外気圏データ読み込みエラー ({time_id}): {e}")
            continue

        total_surface = 0.0

        if PLOT_SURFACE:
            s_filename = f"surface_density_{time_id}.npy"
            s_file = os.path.join(RESULT_DIR, s_filename)

            if os.path.exists(s_file):
                try:
                    s_data = np.load(s_file)
                    total_surface = np.sum(s_data * cell_areas)
                except Exception as e:
                    print(f"表面データ読み込みエラー ({time_id}): {e}")

        # データリストに追加
        data_list.append({
            "taa": taa,
            "surface": total_surface,
            "exosphere": total_exosphere
        })

    if not data_list:
        print("有効なデータが一つも読み込めませんでした。")
        return

    # TAA順にソート
    data_list.sort(key=lambda x: x["taa"])

    taas = np.array([d["taa"] for d in data_list])
    total_surfs = np.array([d["surface"] for d in data_list])
    total_exos = np.array([d["exosphere"] for d in data_list])

    print(f"プロット用データ点数: {len(taas)}")

    # ==========================================================================
    # プロット
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    SCALE_UNIT = 1e28

    # 1. 元のデータ (Raw)
    y_exosphere_raw = total_exos / SCALE_UNIT
    ax.plot(taas, y_exosphere_raw, 'k--', label='Exosphere (Raw)', linewidth=2, marker='x', markersize=4, alpha=0.6)

    # 2. 補正後のデータ (Corrected)
    if APPLY_CORRECTION:
        y_exosphere_corrected = y_exosphere_raw * CORRECTION_FACTOR
        label_text = f'Exosphere (Corrected x{CORRECTION_FACTOR:.2f})'
        ax.plot(taas, y_exosphere_corrected, 'r-', label=label_text, linewidth=2, marker='o', markersize=4)

        print(f"補正を適用しました: 係数 {CORRECTION_FACTOR:.2f}")
        print(f"  - Raw Peak: {np.max(y_exosphere_raw):.2f}")
        print(f"  - Corrected Peak: {np.max(y_exosphere_corrected):.2f} (Target ~4.0?)")

    # 3. 表面データ
    if PLOT_SURFACE:
        y_surface = (total_surfs / 100.0) / SCALE_UNIT
        ax.plot(taas, y_surface, 'k-', label='Surface Na (divided by 100)', linewidth=2, marker='.', markersize=2)

    # 補助線
    ax.axvline(x=23, color='gray', linestyle=':', linewidth=1)
    ax.axvline(x=335, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel("True Anomaly Angle (degrees)", fontsize=14)
    ax.set_ylabel(r"Number of Na $\times 10^{28}$", fontsize=14)
    ax.set_xlim(0, 360)

    # 縦軸の範囲を少し広げる (補正後の値が見切れないように)
    ax.set_ylim(0, max(np.max(y_exosphere_raw), np.max(y_surface)) * 1.8)
    if APPLY_CORRECTION:
        ax.set_ylim(0, max(np.max(y_exosphere_corrected), np.max(y_surface)) * 1.2)

    ax.grid(True, linestyle='-', alpha=0.5)
    ax.legend(fontsize=12, loc='upper center')

    title_text = "Total Number of Na Atoms (with Correction)" if APPLY_CORRECTION else "Total Number of Na Atoms (Raw)"
    plt.title(title_text, fontsize=16)
    plt.tight_layout()

    save_path = "leblanc_fig5_corrected.png"
    plt.savefig(save_path, dpi=300)
    print(f"グラフを保存しました: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()