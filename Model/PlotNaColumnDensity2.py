import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import pandas as pd

# --- 1. 物理定数と正規化因子 ---
RM_m = 2.440e6  # 水星の半径 [m]
CM_PER_M = 100.0
CM2_PER_M2 = CM_PER_M * CM_PER_M

# 観測データ処理で使用している正規化面積 [cm^2]
NORMALIZATION_AREA_CM2 = 3.7408e17
# 半球 (Dawn全体 / Dusk全体)
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0
# 半球の半分 (Dawn外側 / Dusk外側)
NORMALIZATION_AREA_QUARTER_CM2 = NORMALIZATION_AREA_CM2 / 4.0

print(f"正規化面積 (全体): {NORMALIZATION_AREA_CM2:.4e} cm^2")
print(f"正規化面積 (1/2): {NORMALIZATION_AREA_HALF_CM2:.4e} cm^2")
print(f"正規化面積 (1/4): {NORMALIZATION_AREA_QUARTER_CM2:.4e} cm^2")

# --- 2. ユーザー設定 ---

# ★★★ グリッド設定
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★★★ シミュレーション結果のディレクトリ
output_dir = r"./SimulationResult_202511\DynamicGrid72x36_17.0"
#output_dir = r"./SimulationResult_202510/Grid101_Range5RM_SP1e+27_TD"

# ★★★ プロットモード選択
# 選択肢: "ALL", "DAYSIDE_TOTAL", "DAWN", "DUSK", "DAWN_OUTER", "DUSK_OUTER"
PLOT_MODE = "DUSK"

# ★★★ 比較用CSVファイルの設定
SHOW_CSV_OVERLAY = True  # CSVを重ねて表示するか
CSV_PATH = r"./dusk.csv"  # CSVファイルのパス
CSV_LABEL = "Observation"  # 凡例名
CSV_USE_SHARED_Y_AXIS = False  # True: 左軸を共有 / False: 右軸を使用(スケール分離)

# --- 3. グリッド計算準備 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
print(f"グリッド解像度: {GRID_RESOLUTION}x{GRID_RESOLUTION}x{GRID_RESOLUTION}")

mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2
# 1/4領域の境界インデックス
quarter_index_offset = mid_index_y // 2
idx_dawn_outer_limit = quarter_index_offset
idx_dusk_outer_start = (GRID_RESOLUTION - 1) - quarter_index_offset


# --- 4. データ処理関数 ---
def process_simulation_data(target_dir, mode):
    """
    指定ディレクトリのデータを読み込み、指定モードに対応するTAAと密度配列を返す
    """
    sim_results_taa = []
    sim_results_density = []

    try:
        all_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')])
        if not all_files:
            print(f"エラー: ディレクトリ '{target_dir}' に有効な .npy ファイルがありません。")
            return None, None
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{target_dir}' が見つかりません。")
        return None, None

    print(f"処理モード: {mode}")
    print(f"ファイルを処理中... ({len(all_files)}個)")

    for filename in tqdm(all_files):
        try:
            taa = int(filename.split('_taa')[-1].split('.')[0])
        except (ValueError, IndexError):
            continue

        filepath = os.path.join(target_dir, filename)
        density_grid_m3 = np.load(filepath)

        # 1. 昼側 (X >= 0) のみを切り出し
        dayside_grid = density_grid_m3[mid_index_x:, :, :]

        # 2. 原子の総数に変換
        atoms_grid = dayside_grid * cell_volume_m3

        # 3. 境界処理 (X=0 Terminator面は半分)
        atoms_grid[0, :, :] *= 0.5

        # 4. 集計対象の計算
        total_atoms = 0.0
        target_area = 1.0

        if mode == "DAYSIDE_TOTAL":
            total_atoms = np.sum(atoms_grid)
            target_area = NORMALIZATION_AREA_CM2

        elif mode == "DAWN":
            sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
            sum_mid = np.sum(atoms_grid[:, mid_index_y, :])
            total_atoms = sum_dawn + (0.5 * sum_mid)
            target_area = NORMALIZATION_AREA_HALF_CM2

        elif mode == "DUSK":
            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            sum_mid = np.sum(atoms_grid[:, mid_index_y, :])
            total_atoms = sum_dusk + (0.5 * sum_mid)
            target_area = NORMALIZATION_AREA_HALF_CM2

        elif mode == "DAWN_OUTER":
            sum_dawn_outer = np.sum(atoms_grid[:, :idx_dawn_outer_limit, :])
            total_atoms = sum_dawn_outer
            target_area = NORMALIZATION_AREA_QUARTER_CM2

        elif mode == "DUSK_OUTER":
            sum_dusk_outer = np.sum(atoms_grid[:, idx_dusk_outer_start:, :])
            total_atoms = sum_dusk_outer
            target_area = NORMALIZATION_AREA_QUARTER_CM2

        elif mode == "ALL":
            total_atoms = np.sum(atoms_grid)
            target_area = NORMALIZATION_AREA_CM2

        else:
            print(f"不明なモード: {mode}")
            sys.exit()

        col_density = total_atoms / target_area

        sim_results_taa.append(taa)
        sim_results_density.append(col_density)

    sim_results_taa = np.array(sim_results_taa)
    sim_results_density = np.array(sim_results_density)

    sorted_idx = np.argsort(sim_results_taa)
    return sim_results_taa[sorted_idx], sim_results_density[sorted_idx]


# --- 5. メイン処理とプロット ---

# シミュレーションデータの取得
sim_taa, sim_density = process_simulation_data(output_dir, PLOT_MODE)

if sim_taa is not None:
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- シミュレーションデータのプロット (左軸) ---
    color_sim = 'blue'
    ax1.scatter(sim_taa, sim_density, label=f'Sim: {PLOT_MODE}', color=color_sim, alpha=0.8, s=50, zorder=2)
    ax1.plot(sim_taa, sim_density, color=color_sim, alpha=0.4, linestyle='')

    ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=14)
    ax1.set_ylabel(f'Sim Column Density [atoms/cm²]', fontsize=14, color=color_sim)
    ax1.tick_params(axis='y', labelcolor=color_sim)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))

    # --- CSVデータの重ね書き ---
    if SHOW_CSV_OVERLAY:
        if os.path.exists(CSV_PATH):
            print(f"CSVファイルを読み込み中: {CSV_PATH}")
            try:
                # ★修正: encoding='shift_jis' を追加しました
                # 日本語環境(Excel等)で作られたCSVはshift_jis(またはcp932)である場合が多いため
                try:
                    df = pd.read_csv(CSV_PATH, encoding='shift_jis')
                except UnicodeDecodeError:
                    # shift_jisでもダメならcp932(Windows拡張)を試す
                    df = pd.read_csv(CSV_PATH, encoding='cp932')

                # 少なくとも2列あるか確認
                if df.shape[1] >= 4:
                    csv_taa = df.iloc[:, 3].values
                    csv_density = df.iloc[:, 4].values

                    # TAAでソート
                    sort_csv = np.argsort(csv_taa)
                    csv_taa = csv_taa[sort_csv]
                    csv_density = csv_density[sort_csv]

                    color_csv = 'red'

                    if CSV_USE_SHARED_Y_AXIS:
                        # --- 共通のY軸 ---
                        ax1.scatter(csv_taa, csv_density, label=CSV_LABEL, color=color_csv, marker='o', s=50, zorder=3)
                        print("CSVデータを左軸(共通)にプロットしました。")
                    else:
                        # --- 独立したY軸 (右軸) ---
                        ax2 = ax1.twinx()
                        ax2.scatter(csv_taa, csv_density, label=CSV_LABEL, color=color_csv, marker='o', s=50, zorder=3)

                        ax2.set_ylabel(f'{CSV_LABEL} Density', fontsize=14, color=color_csv)
                        ax2.tick_params(axis='y', labelcolor=color_csv)

                        # 凡例の統合
                        lines_1, labels_1 = ax1.get_legend_handles_labels()
                        lines_2, labels_2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)

                        print("CSVデータを右軸(独立)にプロットしました。")
                else:
                    print("エラー: CSVファイルの列数が不足しています。")
            except Exception as e:
                print(f"CSV読み込みエラー: {e}")
                print("※CSVファイルの文字コードまたは形式を確認してください。")
        else:
            print(f"警告: CSVファイルが見つかりません: {CSV_PATH}")

    # 凡例表示（共通軸の場合のみ）
    if not (SHOW_CSV_OVERLAY and not CSV_USE_SHARED_Y_AXIS):
        ax1.legend(loc='upper left', fontsize=12)

    plt.title(f'Simulation Result: {PLOT_MODE} vs TAA', fontsize=16)
    plt.tight_layout()
    plt.show()

else:
    print("データ処理に失敗したため終了します。")