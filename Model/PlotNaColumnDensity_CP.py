import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys

# --- 1. 物理定数と正規化因子 ---
RM_m = 2.440e6  # 水星の半径 [m]
CM_PER_M = 100.0
CM2_PER_M2 = CM_PER_M * CM_PER_M

# 観測データ処理で使用している正規化面積 [cm^2]
NORMALIZATION_AREA_CM2 = 3.7408e17
# 明け方・夕方側で割るための面積 (昼側半球の半分)
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0

print(f"正規化面積 (昼側全体): {NORMALIZATION_AREA_CM2:.4e} cm^2")
print(f"正規化面積 (半球の半分): {NORMALIZATION_AREA_HALF_CM2:.4e} cm^2")

# --- 2. シミュレーション設定 (★ 2セット分設定する) ---

# ★★★ 比較するシミュレーションのグリッド設定 (両者で共通と仮定)
GRID_RESOLUTION = 101  # グリッド解像度
GRID_MAX_RM = 5.0  # グリッドの最大範囲 (水星半径単位)

# --- 設定 1 ---
output_dir_1 = r"./SimulationResult_202510\Grid101_Budget5000_TD"
# ( "DAWN", "DUSK", "DAYSIDE_TOTAL" のいずれか)
plot_mode_1 = "DAYSIDE_TOTAL"
label_1 = "kahen"  # 凡例に表示する名前

# --- 設定 2 ---
output_dir_2 = r"./SimulationResult_202510/Grid101_Range5RM_SP1e+27_TD"
# ( "DAWN", "DUSK", "DAYSIDE_TOTAL" のいずれか)
plot_mode_2 = "DAYSIDE_TOTAL"
label_2 = "kotei"  # 凡例に表示する名前

# --- 3. グリッド計算 (設定に基づいて計算) ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
print(f"グリッド解像度: {GRID_RESOLUTION}x{GRID_RESOLUTION}x{GRID_RESOLUTION}")

# 中心インデックス (座標系: +X=太陽, +Y=夕方)
mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2


# --- 4. ★ データ処理を関数化 ---
def process_simulation_data(output_dir, plot_mode):
    """
    指定されたディレクトリの.npyファイルを処理し、
    指定されたplot_modeのTAAと柱密度を返す関数
    """
    sim_results_dawn = []
    sim_results_dusk = []
    sim_results_dayside = []
    sim_results_taa = []

    try:
        all_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        if not all_files:
            print(f"エラー: ディレクトリ '{output_dir}' に .npy ファイルが見つかりません。")
            return None, None
        else:
            print(f"処理中: {output_dir} (合計 {len(all_files)} ファイル)")
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{output_dir}' が見つかりません。")
        return None, None

    for filename in tqdm(all_files, desc=f"Processing {Path(output_dir).name}"):
        try:
            taa = int(filename.split('_taa')[-1].split('.')[0])
            sim_results_taa.append(taa)
        except (ValueError, IndexError):
            continue

        filepath = os.path.join(output_dir, filename)
        density_grid_m3 = np.load(filepath)

        total_atoms_dawn_dayside = 0.0
        total_atoms_dusk_dayside = 0.0

        for iz in range(GRID_RESOLUTION):
            for ix in range(mid_index_x, GRID_RESOLUTION):  # 昼側 (X>0)
                for iy in range(GRID_RESOLUTION):  # Y軸全域
                    density_in_cell = density_grid_m3[ix, iy, iz]
                    if density_in_cell == 0: continue
                    atoms_in_cell = density_in_cell * cell_volume_m3

                    if ix == mid_index_x:
                        atoms_to_add = 0.5 * atoms_in_cell
                    else:
                        atoms_to_add = atoms_in_cell

                    if iy < mid_index_y:
                        total_atoms_dawn_dayside += atoms_to_add
                    elif iy > mid_index_y:
                        total_atoms_dusk_dayside += atoms_to_add
                    else:
                        total_atoms_dawn_dayside += 0.5 * atoms_to_add
                        total_atoms_dusk_dayside += 0.5 * atoms_to_add

        col_density_dawn = total_atoms_dawn_dayside / NORMALIZATION_AREA_HALF_CM2
        col_density_dusk = total_atoms_dusk_dayside / NORMALIZATION_AREA_HALF_CM2
        total_atoms_dayside = total_atoms_dawn_dayside + total_atoms_dusk_dayside
        col_density_dayside = total_atoms_dayside / NORMALIZATION_AREA_CM2

        sim_results_dawn.append(col_density_dawn)
        sim_results_dusk.append(col_density_dusk)
        sim_results_dayside.append(col_density_dayside)

    if not sim_results_taa:
        print(f"データがありません: {output_dir}")
        return None, None

    # --- plot_mode に応じて返すデータを選択 ---
    data_to_plot = []
    if plot_mode == "DAWN":
        data_to_plot = sim_results_dawn
    elif plot_mode == "DUSK":
        data_to_plot = sim_results_dusk
    elif plot_mode == "DAYSIDE_TOTAL":
        data_to_plot = sim_results_dayside
    else:
        print(f"エラー: 無効な PLOT_MODE '{plot_mode}' が指定されました。")
        return None, None

    # TAAでソート
    sorted_indices = np.argsort(sim_results_taa)
    plot_taa = np.array(sim_results_taa)[sorted_indices]
    plot_density = np.array(data_to_plot)[sorted_indices]

    print(f"処理完了: {output_dir}")
    return plot_taa, plot_density


# --- 5. ★ 2つのシミュレーションを実行 ---
taa_1, density_1 = process_simulation_data(output_dir_1, plot_mode_1)
taa_2, density_2 = process_simulation_data(output_dir_2, plot_mode_2)

# --- 6. グラフプロット (★ 2つのデータを重ねる) ---
if (taa_1 is not None) and (taa_2 is not None):

    plt.figure(figsize=(12, 7))

    # --- データ 1 をプロット ---
    plt.scatter(taa_1, density_1, label=label_1, color='blue', alpha=0.7, s=30)
    print(f"プロット 1: {label_1}")

    # --- データ 2 をプロット ---
    plt.scatter(taa_2, density_2, label=label_2, color='red', alpha=0.7, s=30)
    print(f"プロット 2: {label_2}")

    # --- グラフの共通設定 ---
    plt.xlabel('True Anomaly Angle, degree', fontsize=14)
    plt.ylabel('Column Density, atoms/cm²', fontsize=14)
    plt.title('Simulation Column Density Comparison vs. TAA', fontsize=16)
    plt.legend()  # 凡例を自動的に表示
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()

    print("グラフを表示します...")
    plt.show()

elif (taa_1 is not None):
    print("データ 2 の処理に失敗したため、データ 1 のみプロットします。")
    plt.figure(figsize=(12, 7))
    plt.scatter(taa_1, density_1, label=label_1, color='blue', alpha=0.7, s=30)
    plt.xlabel('True Anomaly Angle, degree', fontsize=14)
    plt.ylabel('Column Density, atoms/cm²', fontsize=14)
    plt.title('Simulation Column Density Comparison vs. TAA', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.show()

elif (taa_2 is not None):
    print("データ 1 の処理に失敗したため、データ 2 のみプロットします。")
    # (データ2のみのプロットコード... 上とほぼ同じなので省略)
    # ...

else:
    print("両方のデータ処理に失敗したため、プロットできませんでした。")