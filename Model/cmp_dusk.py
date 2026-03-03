import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# --- 1. 物理定数と正規化因子 ---
RM_m = 2.440e6  # 水星の半径 [m]
CM_PER_M = 100.0
CM2_PER_M2 = CM_PER_M * CM_PER_M

# 正規化面積 [cm^2]
NORMALIZATION_AREA_CM2 = 3.7408e17
# Dusk領域は半球なので 1/2
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0

# --- 2. ユーザー設定 (ここに比較したい2つのディレクトリを入力) ---

# ★★★ グリッド設定 (両方のシミュレーションで同じである必要があります)
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★★★ 比較する2つのディレクトリパス
# 例: 変更前 vs 変更後、パラメータA vs パラメータB
dir_1_path = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_1228_s1"
dir_1_label = "Sim 1: 1.85 eV"  # グラフの凡例名

dir_2_path = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_1224_s1"
dir_2_label = "Sim 2: 1.4-2.7 eV"  # グラフの凡例名

# --- 3. グリッド計算準備 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3

mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2

print(f"グリッド解像度: {GRID_RESOLUTION}^3")
print(f"Dusk計算用 正規化面積: {NORMALIZATION_AREA_HALF_CM2:.4e} cm^2")


# --- 4. データ抽出関数 (Dusk専用) ---
def get_dusk_data(target_dir, label_name):
    """
    指定ディレクトリからDusk側のカラム密度を計算して返す
    """
    if not os.path.exists(target_dir):
        print(f"エラー: ディレクトリが見つかりません -> {target_dir}")
        return None, None

    all_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')])

    if not all_files:
        print(f"警告: .npyファイルがありません -> {target_dir}")
        return None, None

    print(f"読み込み中 [{label_name}]: {len(all_files)} files...")

    sim_results_taa = []
    sim_results_density = []

    for filename in tqdm(all_files, desc=label_name):
        try:
            # ファイル名からTAAを取得
            taa = int(filename.split('_taa')[-1].split('.')[0])
        except (ValueError, IndexError):
            continue

        filepath = os.path.join(target_dir, filename)

        # 3Dグリッド読み込み [m^-3]
        density_grid_m3 = np.load(filepath)

        # 1. 昼側 (X >= 0) のみを切り出し
        dayside_grid = density_grid_m3[mid_index_x:, :, :]

        # 2. 原子の総数に変換
        atoms_grid = dayside_grid * cell_volume_m3

        # 3. 境界処理 (X=0 Terminator面は半分)
        atoms_grid[0, :, :] *= 0.5

        # 4. Dusk側の集計
        # Y軸の中央(mid_index_y)より大きいインデックスがDusk
        # 中央のセル自体はDawn/Duskの境界なので半分だけ加算する
        sum_dusk_pure = np.sum(atoms_grid[:, mid_index_y + 1:, :])
        sum_mid_line = np.sum(atoms_grid[:, mid_index_y, :])

        total_atoms_dusk = sum_dusk_pure + (0.5 * sum_mid_line)

        # 5. カラム密度計算
        col_density = total_atoms_dusk / NORMALIZATION_AREA_HALF_CM2

        sim_results_taa.append(taa)
        sim_results_density.append(col_density)

    # 配列化とTAA順のソート
    sim_results_taa = np.array(sim_results_taa)
    sim_results_density = np.array(sim_results_density)

    sort_idx = np.argsort(sim_results_taa)
    return sim_results_taa[sort_idx], sim_results_density[sort_idx]


# --- 5. メイン処理とプロット ---

# データの取得
taa1, dens1 = get_dusk_data(dir_1_path, dir_1_label)
taa2, dens2 = get_dusk_data(dir_2_path, dir_2_label)

# プロット作成
fig, ax = plt.subplots(figsize=(10, 6))

# Sim 1 のプロット
if taa1 is not None and len(taa1) > 0:
    ax.plot(taa1, dens1, color='blue', marker='^', linestyle='', alpha=0.8, label=dir_1_label)
else:
    print(f"Sim 1 ({dir_1_label}) のデータがありません。")

# Sim 2 のプロット
if taa2 is not None and len(taa2) > 0:
    ax.plot(taa2, dens2, color='red', marker='v', linestyle='', alpha=0.8, label=dir_2_label)
else:
    print(f"Sim 2 ({dir_2_label}) のデータがありません。")

# 軸ラベルと設定
ax.set_xlabel('True Anomaly Angle (deg)', fontsize=14)
ax.set_ylabel('Dusk Column Density [atoms/cm²]', fontsize=14)
#ax.set_title('Comparison of Dusk Side Density', fontsize=16)

ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 60))

# 凡例を表示
ax.legend(fontsize=12, loc='upper left')

plt.tight_layout()
plt.show()