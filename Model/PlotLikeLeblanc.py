# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob

# ==============================================================================
# 設定 (シミュレーションコードと合わせる)
# ==============================================================================
# 結果が保存されているディレクトリ
RESULT_DIR = r"./SimulationResult_202511/DynamicGrid72x36_14.0"
#RESULT_DIR  = r"./SimulationResult_202511\SubCycle_72x36_2.0"

# 物理定数
RM = 2.440e6  # 水星半径 [m]
PI = np.pi

# グリッド設定 (シミュレーションコードの値と一致させること)
N_LON_FIXED = 72
N_LAT = 36
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0  # コード内の GRID_MAX_RM


# ==============================================================================
# ヘルパー関数
# ==============================================================================
def calculate_cell_areas():
    """表面グリッドの各セルの面積 [m^2] を計算する"""
    lon_edges = np.linspace(-PI, PI, N_LON_FIXED + 1)
    lat_edges = np.linspace(-PI / 2, PI / 2, N_LAT + 1)
    dlon = lon_edges[1] - lon_edges[0]

    # 緯度帯ごとの面積 (球帯の面積公式)
    # Area = R^2 * dphi * (sin(theta2) - sin(theta1))
    lat_term = np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    cell_areas_1d = (RM ** 2) * dlon * lat_term

    # 2次元配列に拡張 (lon, lat)
    # すべての経度で同じ緯度の面積は同じ
    return np.tile(cell_areas_1d, (N_LON_FIXED, 1))


def calculate_voxel_volume():
    """空間グリッドの1ボクセルあたりの体積 [m^3] を計算する"""
    gmin = -GRID_MAX_RM * RM
    gmax = GRID_MAX_RM * RM
    csize = (gmax - gmin) / GRID_RESOLUTION
    return csize ** 3


def extract_taa_from_filename(filename):
    """ファイル名からTAA (True Anomaly Angle) を抽出する"""
    # 想定形式: surface_density_tXXXXX_taaYYY.npy
    match = re.search(r'taa(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    if not os.path.exists(RESULT_DIR):
        print(f"エラー: ディレクトリ '{RESULT_DIR}' が見つかりません。")
        return

    # ファイルリストの取得
    surface_files = glob.glob(os.path.join(RESULT_DIR, "surface_density_*.npy"))

    if not surface_files:
        print("データファイルが見つかりません。シミュレーションは実行されていますか？")
        return

    # データを格納するリスト
    data_list = []

    cell_areas = calculate_cell_areas()
    voxel_volume = calculate_voxel_volume()

    print("データを読み込み中...")

    for s_file in sorted(surface_files):
        taa = extract_taa_from_filename(s_file)
        if taa is None:
            continue

        # 対応する空間密度ファイルを探す
        # surface_density_tXXXXX_taaYYY.npy -> density_grid_tXXXXX_taaYYY.npy
        base_name = os.path.basename(s_file)
        exo_name = base_name.replace("surface_density", "density_grid")
        e_file = os.path.join(RESULT_DIR, exo_name)

        if not os.path.exists(e_file):
            continue

        # --- データの読み込みと積分 ---

        # 1. 表面 (Surface)
        # 単位: [atoms/m^2] -> 総数: sum(density * area)
        s_data = np.load(s_file)
        total_surface = np.sum(s_data * cell_areas)

        # 2. 外気圏 (Exosphere)
        # 単位: [atoms/m^3] -> 総数: sum(density * volume)
        e_data = np.load(e_file)
        total_exosphere = np.sum(e_data) * voxel_volume

        data_list.append({
            "taa": taa,
            "surface": total_surface,
            "exosphere": total_exosphere
        })

    # TAA順にソート
    data_list.sort(key=lambda x: x["taa"])

    # 配列に変換
    taas = np.array([d["taa"] for d in data_list])
    total_surfs = np.array([d["surface"] for d in data_list])
    total_exos = np.array([d["exosphere"] for d in data_list])

    # ==========================================================================
    # プロット (Leblanc 2003 Fig 5 再現)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Y軸のスケール単位 (10^28)
    SCALE_UNIT = 1e28

    # プロットデータ作成
    # 論文では Surface は 1/100 にスケールダウンされている
    y_surface = (total_surfs / 100.0) / SCALE_UNIT
    y_exosphere = total_exos / SCALE_UNIT

    # 1. 表面 (実線)
    ax.plot(taas, y_surface, 'k-', label='Surface Na (divided by 100)', linewidth=2)

    # 2. 外気圏 (破線)
    ax.plot(taas, y_exosphere, 'k--', label='Exosphere Na', linewidth=2)

    # 太陽の回転反転線の描画 (論文にある垂直点線)
    # TAA = 23 deg, 335 deg 付近
    ax.axvline(x=23, color='gray', linestyle=':', linewidth=1)
    ax.axvline(x=335, color='gray', linestyle=':', linewidth=1)

    # 軸ラベルと設定
    ax.set_xlabel("True Anomaly Angle (degrees)", fontsize=14)
    ax.set_ylabel(r"Number of Na $\times 10^{28}$", fontsize=14)
    ax.set_xlim(0, 360)

    # グリッドと凡例
    ax.grid(True, linestyle='-', alpha=0.5)
    ax.legend(fontsize=12, loc='upper center')

    plt.title("Total Number of Na Atoms: Surface vs Exosphere", fontsize=16)
    plt.tight_layout()

    # 保存と表示
    save_path = "leblanc_fig5_reproduction.png"
    plt.savefig(save_path, dpi=300)
    print(f"グラフを保存しました: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()