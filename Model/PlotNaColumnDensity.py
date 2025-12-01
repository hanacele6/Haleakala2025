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
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0

print(f"正規化面積 (昼側全体): {NORMALIZATION_AREA_CM2:.4e} cm^2")
print(f"正規化面積 (半球の半分): {NORMALIZATION_AREA_HALF_CM2:.4e} cm^2")

# --- 2. シミュレーション設定 ---
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★★★ .npy ファイルが保存されているディレクトリパス
output_dir = r"./SimulationResult_202511\DynamicGrid72x36_14.0"
#output_dir = r"./SimulationResult_202511\SubCycle_144x72_4.0"
#output_dir = r"./SimulationResult_202511\Stabilized_v2_Grid72x36_SPS1000"



# プロット選択 ("DAYSIDE_TOTAL", "DAWN", "DUSK")
PLOT_MODE = "DAYSIDE_TOTAL"
#PLOT_MODE = "DUSK"
#PLOT_MODE = "DAWN"
#PLOT_MODE = "ALL"

# --- 3. グリッド計算準備 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
print(f"グリッド解像度: {GRID_RESOLUTION}x{GRID_RESOLUTION}x{GRID_RESOLUTION}")

mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2

# --- 4. ファイルを処理してデータを集計 (高速化版) ---
sim_results_dawn = []
sim_results_dusk = []
sim_results_dayside = []
sim_results_taa = []

try:
    all_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy') and f.startswith('density_grid_')])

    if not all_files:
        print(f"エラー: ディレクトリ '{output_dir}' に .npy ファイルが見つかりません。")
        sys.exit()
    else:
        print(f"合計 {len(all_files)} 個の .npy ファイルを高速処理します...")
except FileNotFoundError:
    print(f"エラー: ディレクトリ '{output_dir}' が見つかりません。パスを確認してください。")
    sys.exit()

for filename in tqdm(all_files, desc="Processing files"):
    # --- ファイル名からTAAを取得 ---
    try:
        taa = int(filename.split('_taa')[-1].split('.')[0])
    except (ValueError, IndexError):
        continue

    # --- 密度グリッドをロード [atoms/m^3] ---
    filepath = os.path.join(output_dir, filename)
    density_grid_m3 = np.load(filepath)

    """
    if taa == 180:  # TAA=180 の時だけ確認
        print(f"\n--- Debug Check for TAA={taa} ---")

        # 1. セルのサイズ（厚み）を計算 [m]
        #    (GRID_MAX_RM と GRID_RESOLUTION はコード上部の設定値を使います)
        grid_width_m = 2 * GRID_MAX_RM * RM_m
        cell_size_m = grid_width_m / GRID_RESOLUTION

        # 2. 柱密度に変換 [atoms/cm^2]
        #    計算式: sum(密度) * 厚み * (1e-4)
        #    axis=2 (Z軸) をつぶして、上から見た図にします
        debug_image = np.sum(density_grid_m3, axis=2) * cell_size_m * 1e-4

        # 3. プロット
        plt.figure(figsize=(6, 5))

        # vmin/vmax も正式なプロットに合わせると比較しやすいです
        plt.imshow(debug_image, origin='lower', cmap='inferno', vmin=1e8, vmax=1e11)

        plt.title(f"Debug View (Correct Units)\nTAA={taa}")
        plt.colorbar(label='Column Density [atoms/cm$^2$]')
        plt.xlabel('Axis 1 (Horizontal)')
        plt.ylabel('Axis 0 (Vertical, Sun Direction)')
        plt.show()

        print(f"最大値: {np.max(debug_image):.2e} atoms/cm^2")
        input("値の桁が正式なプロットと合っているか確認してください...")
    # ▲▲▲ ---------------------------------- ▲▲▲
    """

    # 1. 昼側 (X >= 0) のみを切り出す
    #    index: mid_index_x から 最後まで
    dayside_grid = density_grid_m3[mid_index_x:, :, :]


    # 2. 原子の総数に変換 (密度 * 体積)
    atoms_grid = dayside_grid * cell_volume_m3

    # 3. 境界処理 (X=0 の平面は半分にする)
    #    切り出した配列の 0番目 が X=0 平面 (Terminator)
    atoms_grid[0, :, :] *= 0.5

    # 4. Y軸方向の分割集計
    #    dayside_gridの形状は [X軸(半分), Y軸(全体), Z軸(全体)]
    #    Y軸のインデックスは元の配列と同じ扱いでOK

    # 明け方側 (Y < mid_index_y) の合計
    sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])

    # 夕方側 (Y > mid_index_y) の合計
    sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])

    # Y=0 平面の合計 (半分ずつ分け合う)
    sum_mid_y = np.sum(atoms_grid[:, mid_index_y, :])

    # 合計原子数
    total_atoms_dawn_dayside = sum_dawn + (0.5 * sum_mid_y)
    total_atoms_dusk_dayside = sum_dusk + (0.5 * sum_mid_y)

    # --- 柱密度計算 ---
    col_density_dawn = total_atoms_dawn_dayside / NORMALIZATION_AREA_HALF_CM2
    col_density_dusk = total_atoms_dusk_dayside / NORMALIZATION_AREA_HALF_CM2

    total_atoms_dayside = total_atoms_dawn_dayside + total_atoms_dusk_dayside
    col_density_dayside = total_atoms_dayside / NORMALIZATION_AREA_CM2

    sim_results_taa.append(taa)
    sim_results_dawn.append(col_density_dawn)
    sim_results_dusk.append(col_density_dusk)
    sim_results_dayside.append(col_density_dayside)

print("データ処理が完了しました。")

# --- 7. グラフプロット ---
if sim_results_taa:
    # TAAでソート
    sorted_indices = np.argsort(sim_results_taa)
    plot_taa = np.array(sim_results_taa)[sorted_indices]
    plot_dawn = np.array(sim_results_dawn)[sorted_indices]
    plot_dusk = np.array(sim_results_dusk)[sorted_indices]
    plot_dayside = np.array(sim_results_dayside)[sorted_indices]

    plt.figure(figsize=(12, 7))
    plot_title = ""

    # ★★★ ここに "ALL" モードを追加 ★★★
    if PLOT_MODE == "ALL":
        print("プロットモード: 全データ (ALL)")
        # 3つのデータを重ねてプロット
        plt.scatter(plot_taa, plot_dayside, label='Dayside Total (X>0)', color='green', alpha=0.6, s=40, marker='o')
        plt.scatter(plot_taa, plot_dawn, label='Dawn (Y<0)', color='blue', alpha=0.6, s=30, marker='^')
        plt.scatter(plot_taa, plot_dusk, label='Dusk (Y>0)', color='red', alpha=0.6, s=30, marker='v')

        plot_title = 'Simulation Comparison: Dawn vs Dusk vs Total'

    elif PLOT_MODE == "DAWN":
        plt.scatter(plot_taa, plot_dawn, label='Simulation: Dawn Dayside (Y<0, X>0)', color='blue', alpha=0.7, s=30)
        plot_title = 'Simulation (Dawn Side) vs. TAA'
        print("プロットモード: 明け方 (DAWN)")

    elif PLOT_MODE == "DUSK":
        plt.scatter(plot_taa, plot_dusk, label='Simulation: Dusk Dayside (Y>0, X>0)', color='red', alpha=0.7, s=30)
        plot_title = 'Simulation (Dusk Side) vs. TAA'
        print("プロットモード: 夕方 (DUSK)")

    elif PLOT_MODE == "DAYSIDE_TOTAL":
        plt.scatter(plot_taa, plot_dayside, label='Simulation: Total Dayside (X>0)', color='green', alpha=0.7, s=30)
        plot_title = 'Simulation (Total Dayside) vs. TAA'
        print("プロットモード: 昼側全体 (DAYSIDE_TOTAL)")

    else:
        print(f"エラー: PLOT_MODE '{PLOT_MODE}' は無効です。")
        sys.exit()

    plt.xlabel('True Anomaly Angle, degree', fontsize=14)
    plt.ylabel('Column Density, atoms/cm²', fontsize=14)
    plt.title(plot_title, fontsize=16)
    plt.legend(fontsize=12)  # 凡例を見やすく
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()

    print("グラフを表示します...")
    plt.show()

else:
    print("プロットするデータがありませんでした。output_dir のパスを確認してください。")