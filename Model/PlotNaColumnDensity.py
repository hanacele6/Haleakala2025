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
# (水星の昼側表面積: 2 * pi * (2.44e8 cm)^2)
NORMALIZATION_AREA_CM2 = 3.7408e17
# ★ 変更点 1: 明け方・夕方側で割るための面積 (昼側半球の半分)
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0

print(f"正規化面積 (昼側全体): {NORMALIZATION_AREA_CM2:.4e} cm^2")
print(f"正規化面積 (半球の半分): {NORMALIZATION_AREA_HALF_CM2:.4e} cm^2")

# --- 2. シミュレーション設定 (ご自身のコードから転記) ---
GRID_RESOLUTION = 101  # グリッド解像度
GRID_MAX_RM = 5.0  # グリッドの最大範囲 (水星半径単位)
# ★★★ .npy ファイルが保存されているディレクトリパス
output_dir = r"./SimulationResult_202510/Grid101_Range5RM_SP5e+21_SWS_COSOFF"
#output_dir = r"./SimulationResult_202510\Grid101_Budget5000_TD"

# ★ 変更点 2: プロット選択
# "DAYSIDE_TOTAL", "DAWN", "DUSK" のいずれかを指定
PLOT_MODE = "DAYSIDE_TOTAL"

# --- 3. グリッド計算 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
print(f"グリッド解像度: {GRID_RESOLUTION}x{GRID_RESOLUTION}x{GRID_RESOLUTION}")

# 中心インデックス (座標系: +X=太陽, +Y=夕方)
mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2

# --- 4. ファイルを処理してデータを集計 ---
sim_results_dawn = []
sim_results_dusk = []
sim_results_dayside = []  # ★ 昼側全体のリストを追加
sim_results_taa = []

try:
    all_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
    if not all_files:
        print(f"エラー: ディレクトリ '{output_dir}' に .npy ファイルが見つかりません。")
        sys.exit()
    else:
        print(f"合計 {len(all_files)} 個の .npy ファイルを処理します...")
except FileNotFoundError:
    print(f"エラー: ディレクトリ '{output_dir}' が見つかりません。パスを確認してください。")
    sys.exit()

for filename in tqdm(all_files, desc="Processing files"):
    # --- ファイル名からTAAを取得 ---
    try:
        taa = int(filename.split('_taa')[-1].split('.')[0])
        sim_results_taa.append(taa)
    except (ValueError, IndexError):
        continue

    # --- 密度グリッドをロード [atoms/m^3] ---
    filepath = os.path.join(output_dir, filename)
    density_grid_m3 = np.load(filepath)

    total_atoms_dawn_dayside = 0.0
    total_atoms_dusk_dayside = 0.0

    # 3Dグリッドを全走査
    for iz in range(GRID_RESOLUTION):
        for ix in range(mid_index_x, GRID_RESOLUTION):  # 昼側 (X>0)
            for iy in range(GRID_RESOLUTION):  # Y軸全域

                density_in_cell = density_grid_m3[ix, iy, iz]
                if density_in_cell == 0:
                    continue

                atoms_in_cell = density_in_cell * cell_volume_m3

                # --- X軸の境界処理 (X=0 の平面) ---
                if ix == mid_index_x:
                    atoms_to_add = 0.5 * atoms_in_cell
                else:
                    atoms_to_add = atoms_in_cell

                # --- Y軸の境界処理 (明け方/夕方) ---
                if iy < mid_index_y:
                    total_atoms_dawn_dayside += atoms_to_add
                elif iy > mid_index_y:
                    total_atoms_dusk_dayside += atoms_to_add
                else:  # Y = 0 の平面
                    total_atoms_dawn_dayside += 0.5 * atoms_to_add
                    total_atoms_dusk_dayside += 0.5 * atoms_to_add

    # --- 6. 柱密度を計算 (観測の正規化方法に合わせる) ---

    # ★ 変更点 1: 明け方と夕方は「半分の面積」で割る
    col_density_dawn = total_atoms_dawn_dayside / NORMALIZATION_AREA_HALF_CM2
    col_density_dusk = total_atoms_dusk_dayside / NORMALIZATION_AREA_HALF_CM2

    # ★ 追加: 昼側全体は「全体の面積」で割る
    total_atoms_dayside = total_atoms_dawn_dayside + total_atoms_dusk_dayside
    col_density_dayside = total_atoms_dayside / NORMALIZATION_AREA_CM2

    sim_results_dawn.append(col_density_dawn)
    sim_results_dusk.append(col_density_dusk)
    sim_results_dayside.append(col_density_dayside)  # ★ リストに追加

print("データ処理が完了しました。")

# --- 7. グラフプロット (plt.show() を使用) ---
if sim_results_taa:
    # TAAでソート
    sorted_indices = np.argsort(sim_results_taa)
    plot_taa = np.array(sim_results_taa)[sorted_indices]
    plot_dawn = np.array(sim_results_dawn)[sorted_indices]
    plot_dusk = np.array(sim_results_dusk)[sorted_indices]
    plot_dayside = np.array(sim_results_dayside)[sorted_indices]  # ★ ソート

    plt.figure(figsize=(12, 7))
    plot_title = ""

    # ★ 変更点 2: PLOT_MODE に応じて描画内容を変更

    if PLOT_MODE == "DAWN":
        plt.scatter(plot_taa, plot_dawn, label='Simulation: Dawn Dayside (Y<0, X>0)', color='blue', alpha=0.7, s=30)
        plot_title = 'Simulation (Normalized by Half Dayside Area) vs. TAA'
        print("プロットモード: 明け方 (DAWN)")

    elif PLOT_MODE == "DUSK":
        plt.scatter(plot_taa, plot_dusk, label='Simulation: Dusk Dayside (Y>0, X>0)', color='red', alpha=0.7, s=30)
        plot_title = 'Simulation (Normalized by Half Dayside Area) vs. TAA'
        print("プロットモード: 夕方 (DUSK)")

    elif PLOT_MODE == "DAYSIDE_TOTAL":
        plt.scatter(plot_taa, plot_dayside, label='Simulation: Total Dayside (X>0)', color='green', alpha=0.7, s=30)
        plot_title = 'Simulation vs. TAA'
        print("プロットモード: 昼側全体 (DAYSIDE_TOTAL)")

    else:
        print(
            f"エラー: PLOT_MODE '{PLOT_MODE}' は無効です。'DAWN', 'DUSK', 'DAYSIDE_TOTAL' のいずれかを指定してください。")
        sys.exit()

    plt.xlabel('True Anomaly Angle, degree', fontsize=14)
    plt.ylabel('Column Density, atoms/cm²', fontsize=14)
    plt.title(plot_title, fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # 横軸の範囲を 0 から 360 に固定
    plt.xlim(0, 360)
    # 横軸の目盛りを 60 刻みに設定 (0, 60, 120, ..., 360)
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()

    print("グラフを表示します...")
    plt.show()

else:
    print("プロットするデータがありませんでした。output_dir のパスを確認してください。")