import numpy as np
import os
import glob
import re
from tqdm import tqdm

# --- ★★★ 設定項目 ★★★ ---
RESULTS_DIR = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D"
FILE_PATTERN = "density3d_map_taa*_beta0.50_RA_pl24x24.npy"
OUTPUT_FILE = "precomputed_column_densities.npy"

# --- 計算範囲と定数 (解析コードと一致させる) ---
RADIAL_RANGE_RM_MIN = 1.0
RADIAL_RANGE_RM_MAX = 5.0  # グリッド全体を対象にする
GRID_RADIUS_RM = 5.0
RM_m = 2439.7e3


# --- ここまで設定項目 ---

def precompute_all_densities():
    """
    全てのTAAファイルに対し、表面の全地点の柱密度を計算して一つのファイルに保存する。
    """
    search_path = os.path.join(RESULTS_DIR, '**', FILE_PATTERN)
    file_list = glob.glob(search_path, recursive=True)

    if not file_list:
        print(f"エラー: '{search_path}' に一致するファイルが見つかりません。")
        return

    print(f"{len(file_list)}個のファイルを検出しました。事前計算を開始します...")

    # TAAとファイルパスのペアを作成し、TAAでソート
    taa_files = []
    for filepath in file_list:
        match = re.search(r'taa(\d+)', os.path.basename(filepath))
        if match:
            taa_files.append((int(match.group(1)), filepath))

    taa_files.sort()

    # 最初のファイルからグリッド形状を取得
    temp_grid = np.load(taa_files[0][1])
    N_R, N_THETA, N_PHI = temp_grid.shape

    # 結果を格納する配列 (TAAの数, N_THETA, N_PHI)
    num_taas = len(taa_files)
    all_column_densities = np.zeros((num_taas, N_THETA, N_PHI))

    # 定数計算
    DR_m = (RM_m * GRID_RADIUS_RM) / N_R
    DR_cm = DR_m * 100
    ir_start = int((RADIAL_RANGE_RM_MIN / GRID_RADIUS_RM) * N_R)
    ir_end = int((RADIAL_RANGE_RM_MAX / GRID_RADIUS_RM) * N_R)

    for i, (taa, filepath) in enumerate(tqdm(taa_files, desc="Processing TAA files")):
        number_density_grid_cm3 = np.load(filepath)

        # 半径方向に積分して柱密度を計算 [atoms/cm^2]
        # この操作で (N_R, N_THETA, N_PHI) -> (N_THETA, N_PHI) の2D配列になる
        column_density_grid = np.sum(number_density_grid_cm3[ir_start:ir_end, :, :], axis=0) * DR_cm

        all_column_densities[i, :, :] = column_density_grid

    np.save(OUTPUT_FILE, all_column_densities)
    # TAAのリストも保存しておくと便利
    taas = [item[0] for item in taa_files]
    np.save("taa_axis.npy", np.array(taas))

    print(f"\n事前計算が完了し、{OUTPUT_FILE} と taa_axis.npy に保存しました。")
    print(f"データ形状: {all_column_densities.shape} (TAA, Theta, Phi)")


if __name__ == '__main__':
    precompute_all_densities()