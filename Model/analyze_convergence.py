import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ==============================================================================
# 1. ユーザー設定 (比較したい2つのディレクトリを指定)
# ==============================================================================

# ★ 比較元 (例: スピンアップ2年)
dir_run_a = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0116_0.4Denabled_2.7_LowestQ"

# ★ 比較先 (例: スピンアップ3年)
dir_run_b = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0119_0.4Denabled_2.7_LowestQ+1spin"

# ラベル
label_a = "SpinUp 2yr"
label_b = "SpinUp 3yr"

# ★ 下段の差分パネルを表示するか (True: 表示 / False: 非表示)
SHOW_DIFF_PANEL = False

# 物理定数・グリッド設定
RM_m = 2.440e6
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★ 正規化面積 (全球昼側)
NORMALIZATION_AREA_CM2 = 3.7408e17

# ==============================================================================
# 2. データ処理関数 (DAYSIDE_TOTAL専用)
# ==============================================================================
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
mid_index_x = (GRID_RESOLUTION - 1) // 2


def load_dayside_total(target_dir, label):
    if not os.path.exists(target_dir):
        print(f"Error: Directory not found: {target_dir}")
        return None, None

    all_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')])
    print(f"[{label}] Loading {len(all_files)} files...")

    taas = []
    densities = []

    for filename in tqdm(all_files, leave=False):
        try:
            t = int(filename.split('_taa')[-1].split('.')[0])
        except:
            continue

        filepath = os.path.join(target_dir, filename)
        # グリッド読み込み
        grid = np.load(filepath)

        # 1. 昼側 (X >= 0) を切り出し
        dayside_grid = grid[mid_index_x:, :, :]

        # 2. 粒子数に変換
        atoms_grid = dayside_grid * cell_volume_m3

        # 3. Terminator面 (X=0) の補正 (半分だけカウント)
        atoms_grid[0, :, :] *= 0.5

        # 4. 全合計 (Dayside Total Atoms)
        total_atoms = np.sum(atoms_grid)

        # 5. カラム密度へ変換 (全原子数 / 全面積)
        col_dens = total_atoms / NORMALIZATION_AREA_CM2

        taas.append(t)
        densities.append(col_dens)

    # ソート
    taas = np.array(taas)
    densities = np.array(densities)
    idx = np.argsort(taas)

    return taas[idx], densities[idx]


# ==============================================================================
# 3. メイン処理: 比較とプロット
# ==============================================================================

# データ読み込み
taa_a, dens_a = load_dayside_total(dir_run_a, label_a)
taa_b, dens_b = load_dayside_total(dir_run_b, label_b)

if taa_a is None or taa_b is None:
    exit()

# 共通TAAでのマッチング
common_taa = np.intersect1d(taa_a, taa_b)
val_a = []
val_b = []

for t in common_taa:
    val_a.append(dens_a[np.where(taa_a == t)[0][0]])
    val_b.append(dens_b[np.where(taa_b == t)[0][0]])

val_a = np.array(val_a)
val_b = np.array(val_b)

# --- 誤差計算 ---
diff = val_b - val_a
rmse = np.sqrt(np.mean(diff ** 2))
avg_val = (val_a + val_b) / 2.0
rel_diff_percent = np.mean(np.abs(diff) / avg_val) * 100.0

print(f"\n=== Convergence Check (Dayside Total) ===")
print(f"RMSE: {rmse:.2e} atoms/cm^2")
print(f"Mean Relative Difference: {rel_diff_percent:.2f} %")

# --- プロット作成 ---
if SHOW_DIFF_PANEL:
    # 差分あり: 2段組み
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
else:
    # 差分なし: 1段のみ
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax2 = None

# 上段: 値の比較
ax1.plot(taa_a, dens_a, color='b', linestyle='None', label=label_a, alpha=0.6, marker='^', markersize=4)
ax1.plot(taa_b, dens_b, color='r', linestyle='None', label=label_b, alpha=0.8, marker='.', markersize=4)

# テキスト表示
#stats_text = f"RMSE: {rmse:.1e}\nDiff: {rel_diff_percent:.1f}%"
#ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
#         fontsize=12, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')

ax1.set_ylabel("Dayside Total Density\n[atoms/cm²]", fontsize=14)
#ax1.set_title("Spin-Up Convergence: Dayside Total", fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.5)

# 下段: 差分 (SHOW_DIFF_PANEL が True の場合のみ描画)
if SHOW_DIFF_PANEL and ax2 is not None:
    ax2.plot(common_taa, diff, 'k.-', label=f"Diff ({label_b} - {label_a})")
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    ax2.set_ylabel("Difference", fontsize=12)
    ax2.set_xlabel("True Anomaly Angle (deg)", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
else:
    # 下段がない場合は上段にX軸ラベルをつける
    ax1.set_xlabel("True Anomaly Angle (deg)", fontsize=14)

plt.tight_layout()
plt.show()