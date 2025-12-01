import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import pandas as pd

# --- 1. 物理定数と正規化因子 ---
RM_m = 2.440e6
CM_PER_M = 100.0
NORMALIZATION_AREA_CM2 = 3.7408e17
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0

# --- 2. シミュレーション設定 ---
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

output_dir = r"./SimulationResult_202510\DynamicGrid72x36_7.0"
OBS_DATA_PATH = r"./dusk.csv"
PLOT_MODE = "DAYSIDE_TOTAL"  # "DAYSIDE_TOTAL", "DAWN", "DUSK"

# --- 3. グリッド計算準備 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3

mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2

# --- 4. データ処理 (高速化版) ---
sim_results_dawn = []
sim_results_dusk = []
sim_results_dayside = []
sim_results_taa = []

try:
    all_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy') and f.startswith('density_grid_')])
    if not all_files:
        sys.exit(f"エラー: {output_dir} に .npy ファイルがありません。")
except FileNotFoundError:
    sys.exit(f"エラー: ディレクトリ {output_dir} が見つかりません。")

print(f"{len(all_files)} 個のファイルを高速処理します...")

for filename in tqdm(all_files, desc="Processing"):
    try:
        taa = int(filename.split('_taa')[-1].split('.')[0])
    except (ValueError, IndexError):
        continue

    filepath = os.path.join(output_dir, filename)
    density_grid_m3 = np.load(filepath)

    # --- ★★★ 高速化ポイント: forループを全廃し、Numpyスライス計算に変更 ---

    # 1. 昼側 (X >= 0) のみを切り出す
    #    index: mid_index_x から 最後まで
    dayside_grid = density_grid_m3[mid_index_x:, :, :]

    # 2. 原子の総数に変換 (密度 * 体積)
    atoms_grid = dayside_grid * cell_volume_m3

    # 3. 境界処理 (X=0 の平面は半分にする)
    #    切り出した配列の 0番目 が X=0 平面
    atoms_grid[0, :, :] *= 0.5

    # 4. Y軸方向の分割集計
    #    dayside_gridの形状は [X軸(半分), Y軸(全体), Z軸(全体)]
    #    Y軸のインデックスは元の配列と同じ扱いでOK

    # 明け方側 (Y < 0) の合計
    sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])

    # 夕方側 (Y > 0) の合計
    sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])

    # Y=0 平面の合計 (半分ずつ分け合う)
    sum_mid_y = np.sum(atoms_grid[:, mid_index_y, :])

    total_atoms_dawn = sum_dawn + (0.5 * sum_mid_y)
    total_atoms_dusk = sum_dusk + (0.5 * sum_mid_y)

    # --- 柱密度計算 ---
    col_density_dawn = total_atoms_dawn / NORMALIZATION_AREA_HALF_CM2
    col_density_dusk = total_atoms_dusk / NORMALIZATION_AREA_HALF_CM2

    # 昼側全体
    total_atoms_dayside = total_atoms_dawn + total_atoms_dusk
    col_density_dayside = total_atoms_dayside / NORMALIZATION_AREA_CM2

    sim_results_taa.append(taa)
    sim_results_dawn.append(col_density_dawn)
    sim_results_dusk.append(col_density_dusk)
    sim_results_dayside.append(col_density_dayside)

print("データ処理完了。")

# --- 7. グラフプロット ---
if sim_results_taa:
    sorted_indices = np.argsort(sim_results_taa)
    plot_taa = np.array(sim_results_taa)[sorted_indices]
    plot_dawn = np.array(sim_results_dawn)[sorted_indices]
    plot_dusk = np.array(sim_results_dusk)[sorted_indices]
    plot_dayside = np.array(sim_results_dayside)[sorted_indices]

    plt.figure(figsize=(12, 7))

    sim_max = 0  # 初期化

    if PLOT_MODE == "DAWN":
        plt.scatter(plot_taa, plot_dawn, label='Simulation: Dawn', color='blue', alpha=0.7, s=30)
        sim_max = np.max(plot_dawn)
        plot_title = 'Dawn Side Comparison'
    elif PLOT_MODE == "DUSK":
        plt.scatter(plot_taa, plot_dusk, label='Simulation: Dusk', color='red', alpha=0.7, s=30)
        sim_max = np.max(plot_dusk)
        plot_title = 'Dusk Side Comparison'
    elif PLOT_MODE == "DAYSIDE_TOTAL":
        plt.scatter(plot_taa, plot_dayside, label='Simulation: Dayside Total', color='green', alpha=0.7, s=30)
        sim_max = np.max(plot_dayside)
        plot_title = 'Dayside Total Comparison'

    # --- ★ 観測データの読み込み（インデント修正済み） ---
    if os.path.exists(OBS_DATA_PATH):
        try:
            print(f"観測データを読み込んでいます: {OBS_DATA_PATH}")
            try:
                df_obs = pd.read_csv(OBS_DATA_PATH, encoding='shift_jis')
            except UnicodeDecodeError:
                try:
                    df_obs = pd.read_csv(OBS_DATA_PATH, encoding='cp932')
                except UnicodeDecodeError:
                    df_obs = pd.read_csv(OBS_DATA_PATH, encoding='utf-8')

            obs_taa = pd.to_numeric(df_obs.iloc[:, 3], errors='coerce')
            obs_density = pd.to_numeric(df_obs.iloc[:, 4], errors='coerce')

            valid_mask = ~np.isnan(obs_taa) & ~np.isnan(obs_density)
            obs_taa = obs_taa[valid_mask]
            obs_density = obs_density[valid_mask]

            print("-" * 40)
            print(f"【データ確認】")
            if len(obs_taa) > 0:
                print(f"  観測値 TAA範囲: {obs_taa.min():.2f} ~ {obs_taa.max():.2f}")
                print(f"  観測値 密度範囲: {obs_density.min():.2e} ~ {obs_density.max():.2e}")

                plt.scatter(obs_taa, obs_density, label='Observation', color='black', marker='x', s=60, zorder=10)
                print("観測データをプロットしました。")
            else:
                print("警告: 有効なデータがありません。")
            print("-" * 40)

        except Exception as e:
            print(f"観測データの読み込みエラー: {e}")
    else:
        print(f"警告: ファイルが見つかりません {OBS_DATA_PATH}")

    plt.xlabel('True Anomaly Angle (deg)', fontsize=14)
    plt.ylabel('Column Density (atoms/cm²)', fontsize=14)
    plt.title(f"{plot_title} (Sim vs Obs)", fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 60))
    plt.tight_layout()
    plt.show()

else:
    print("データがありません。")