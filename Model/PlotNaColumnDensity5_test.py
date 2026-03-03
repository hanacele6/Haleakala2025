import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import pandas as pd

# =========================================================
# 1. 物理定数と正規化因子
# =========================================================
RM_m = 2.440e6
RM_cm = RM_m * 100.0
CM_PER_M = 100.0
CM2_PER_M2 = CM_PER_M * CM_PER_M

# --- 面積計算 ---
AREA_FULL_DISK = np.pi * RM_cm ** 2


def calculate_segment_area(radius_cm, h_ratio):
    theta = 2.0 * np.arccos(h_ratio)
    area_segment = (radius_cm ** 2 / 2.0) * (theta - np.sin(theta))
    return area_segment


AREA_NORTH = calculate_segment_area(RM_cm, 0.3)
AREA_SOUTH = AREA_NORTH
AREA_EQUATOR_TOTAL = AREA_FULL_DISK - (AREA_NORTH + AREA_SOUTH)

# --- Milillo et al. (2021) 赤道内部の分割面積 (近似) ---
# 赤道領域 (-0.3Rm < Z < 0.3Rm) をさらに経度20度で分割
# 太陽から見た投影面(Y-Z平面)での面積を計算します
# SS領域の幅: Y = +/- Rm * sin(20deg)
SIN_20 = np.sin(np.deg2rad(20))
Y_BOUND_20 = RM_cm * SIN_20

# 近似計算: 赤道領域は長方形に近いと仮定して比率で分割
# (厳密には円の曲線がありますが、赤道付近なので誤差は小さい)
# 全赤道幅(直径)に対するSS領域の幅の比率ではありません。
# 投影面上の幅: 全体 2*Rm に対して、SSは 2*Rm*sin(20)。
# しかし、Equator領域は Zカットされているため、積分範囲での面積比を使うのが適切。
# ここでは簡易的に「グリッド上でカウントしたピクセル数」を面積として使う動的な方法を採用します（後述の関数内で計算）。

# --- 解析モデル用定数 ---
G = 6.67384e-5
MS = 1.9884e30
L_AU = 0.37078
AU = 1.495978707e13
E = 0.2056

# =========================================================
# 2. ユーザー設定
# =========================================================
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ディレクトリ設定 (適宜書き換えてください)
output_dir_1 = r"./SimulationResult_202602/ParabolicHop_72x36_NoEq_DT100_0211_0.4Denabled_2.7_LowestQ_Bounce525K"
label_sim_1 = "Current Model"
ENABLE_SECOND_SIM = False
output_dir_2 = r"./SimulationResult_Old"
label_sim_2 = "Old Model"

# ★ プロットモード: "MILILLO_DETAILED" を追加
# MILILLO: North/Equator/South
# MILILLO_DETAILED: Equator内部をさらに Dawn/SS/Dusk に分割して表示
PLOT_MODE = "MILILLO_DETAILED"

COMMON_Y_LABEL = "Column Density [atoms/cm²]"

# --- 3. グリッド計算準備 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
cell_area_cm2 = (cell_size_m * 100) ** 2  # 投影面の1セルあたりの面積

mid_index = (GRID_RESOLUTION - 1) // 2

# 座標グリッドの作成 (3Dマスク用)
# x: Sun-Planet, y: Dawn-Dusk, z: North-South
x = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, GRID_RESOLUTION)
y = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, GRID_RESOLUTION)
z = np.linspace(-GRID_MAX_RM, GRID_MAX_RM, GRID_RESOLUTION)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# 領域マスクの作成 (一度だけ計算)
# 1. 赤道マスク (-0.3 < z < 0.3)
mask_equator_z = (Z > -0.3) & (Z < 0.3)

# 2. 経度マスク (赤道面内での角度)
# phi = 0 がSub-solar (X軸), phi = 90がDusk (Y軸) と仮定
# 注意: Y軸がDawnかDuskかはシミュレーション座標系に依存します。
# 通常、公転方向が+YならDusk。ここでは +Y = Dusk と仮定します。
Phi = np.degrees(np.arctan2(Y, X))

mask_ss_phi = (np.abs(Phi) <= 20)
mask_dusk_phi = (Phi > 20) & (Phi < 160)  # 裏側を含まないように制限
mask_dawn_phi = (Phi < -20) & (Phi > -160)

# 3. 結合マスク (3D空間上の領域)
# ※ dayside (X>0) はデータロード時にカットするため、ここではY,Z条件のみ定義
# しかし、arctan2を使うためXも必要。ロード後のデータ形状に合わせてスライスして適用します。

print(f"Grid Setup Done. Resolution: {GRID_RESOLUTION}")


# --- 5. データ処理関数 ---
def process_simulation_data(target_dir, mode):
    all_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')])
    if not all_files:
        print("No files found.")
        return None, None

    sim_results_taa = []

    # 格納用辞書の初期化
    if mode == "MILILLO_DETAILED":
        # 赤道を3分割 + 南北
        results_dict = {"NORTH": [], "SOUTH": [],
                        "EQ_DAWN": [], "EQ_SS": [], "EQ_DUSK": []}
    elif mode == "MILILLO":
        results_dict = {"NORTH": [], "EQUATOR": [], "SOUTH": []}
    else:
        results_dict = {"TOTAL": []}

    # 投影面積の計算 (ピクセルカウント方式)
    # 視線方向(X軸)から見たY-Z平面上のピクセル数をカウントして面積とする
    # X軸方向には「1つでも有効なボクセルがあれば」投影面にカウント
    # 実際には「領域内の体積積分」÷「領域の投影面積」＝ 柱密度

    # マスクを昼側半分(X >= 0)に切り出し
    dayside_slice = slice(mid_index, None)
    X_day = X[dayside_slice, :, :]
    Y_day = Y[dayside_slice, :, :]
    Z_day = Z[dayside_slice, :, :]
    Phi_day = Phi[dayside_slice, :, :]

    # 各領域の論理マスク (3D)
    # North/South
    m_north = (Z_day > 0.3)
    m_south = (Z_day < -0.3)
    # Equator全体
    m_eq_all = (Z_day >= -0.3) & (Z_day <= 0.3)

    # Equator細分化
    m_eq_ss = m_eq_all & (np.abs(Phi_day) <= 20)
    m_eq_dusk = m_eq_all & (Phi_day > 20)
    m_eq_dawn = m_eq_all & (Phi_day < -20)

    # 投影面積 (Projected Area) の計算 [cm^2]
    # 方法: マスクされた領域をX軸方向に射影(maxをとる)し、Trueになったピクセル数 * 1セルの面積
    def get_projected_area(mask_3d):
        projection = np.any(mask_3d, axis=0)  # X軸方向に潰す
        return np.sum(projection) * cell_area_cm2

    area_north_calc = get_projected_area(m_north)
    area_south_calc = get_projected_area(m_south)
    area_eq_total_calc = get_projected_area(m_eq_all)
    area_eq_ss_calc = get_projected_area(m_eq_ss)
    area_eq_dawn_calc = get_projected_area(m_eq_dawn)
    area_eq_dusk_calc = get_projected_area(m_eq_dusk)

    # 面積が0の場合のゼロ除算防止
    area_eq_ss_calc = max(area_eq_ss_calc, 1.0)
    area_eq_dawn_calc = max(area_eq_dawn_calc, 1.0)
    area_eq_dusk_calc = max(area_eq_dusk_calc, 1.0)

    print(f" Calculated Areas [cm^2]:")
    print(f"  North: {area_north_calc:.2e}, South: {area_south_calc:.2e}")
    print(f"  Eq_SS: {area_eq_ss_calc:.2e}, Eq_Dawn: {area_eq_dawn_calc:.2e}, Eq_Dusk: {area_eq_dusk_calc:.2e}")

    for filename in tqdm(all_files):
        try:
            taa = int(filename.split('_taa')[-1].split('.')[0])
        except:
            continue

        filepath = os.path.join(target_dir, filename)
        density_grid_m3 = np.load(filepath)

        # 昼側切り出し & 原子数への変換
        dayside_grid = density_grid_m3[dayside_slice, :, :]
        atoms_grid = dayside_grid * cell_volume_m3
        atoms_grid[0, :, :] *= 0.5  # ターミネーター補正

        if mode == "MILILLO_DETAILED":
            # 各領域の総原子数をマスクを使って集計
            n_north = np.sum(atoms_grid[m_north])
            n_south = np.sum(atoms_grid[m_south])
            n_eq_dawn = np.sum(atoms_grid[m_eq_dawn])
            n_eq_ss = np.sum(atoms_grid[m_eq_ss])
            n_eq_dusk = np.sum(atoms_grid[m_eq_dusk])

            results_dict["NORTH"].append(n_north / area_north_calc)
            results_dict["SOUTH"].append(n_south / area_south_calc)
            results_dict["EQ_DAWN"].append(n_eq_dawn / area_eq_dawn_calc)
            results_dict["EQ_SS"].append(n_eq_ss / area_eq_ss_calc)
            results_dict["EQ_DUSK"].append(n_eq_dusk / area_eq_dusk_calc)

        elif mode == "MILILLO":
            n_north = np.sum(atoms_grid[m_north])
            n_south = np.sum(atoms_grid[m_south])
            n_eq = np.sum(atoms_grid[m_eq_all])

            results_dict["NORTH"].append(n_north / area_north_calc)
            results_dict["SOUTH"].append(n_south / area_south_calc)
            results_dict["EQUATOR"].append(n_eq / area_eq_total_calc)

        sim_results_taa.append(taa)

    # ソート
    sim_results_taa = np.array(sim_results_taa)
    idx = np.argsort(sim_results_taa)

    final_dict = {k: np.array(v)[idx] for k, v in results_dict.items()}
    return sim_results_taa[idx], final_dict


# --- 6. 実行とプロット ---
sim_taa, sim_data = process_simulation_data(output_dir_1, PLOT_MODE)

if sim_taa is not None:
    if PLOT_MODE == "MILILLO_DETAILED":
        # 3つのパネルを作成: (North/South), (Equator Total), (Equator Dawn/SS/Dusk)
        # あるいは Milillo Fig 3のように3段にするか、Fig 5のようにEquator詳細だけ見るか

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Panel 1: North / South
        ax = axes[0]
        ax.plot(sim_taa, sim_data["NORTH"], 'b-o', label="North (>0.3Rm)")
        ax.plot(sim_taa, sim_data["SOUTH"], 'r-o', label="South (<-0.3Rm)")
        ax.set_title("Latitudinal Distribution")
        ax.set_ylabel(COMMON_Y_LABEL)
        ax.legend()
        ax.grid(True)

        # Panel 2: Equator Detailed (Dawn/SS/Dusk) -> これが一番見たいもの
        ax = axes[1]
        ax.plot(sim_taa, sim_data["EQ_DAWN"], 'g-^', label="Eq Dawn (<-20°)")
        ax.plot(sim_taa, sim_data["EQ_SS"], 'k-s', label="Eq Sub-Solar (+/-20°)")
        ax.plot(sim_taa, sim_data["EQ_DUSK"], 'm-v', label="Eq Dusk (>20°)")
        ax.set_title("Equatorial Longitudinal Distribution (Milillo et al. 2021)")
        ax.set_ylabel(COMMON_Y_LABEL)
        ax.legend()
        ax.grid(True)

        # Panel 3: Difference (Dawn - SS) etc. (Milillo Fig 5 bottom like)
        ax = axes[2]
        diff_dawn = sim_data["EQ_DAWN"] - sim_data["EQ_SS"]
        diff_dusk = sim_data["EQ_DUSK"] - sim_data["EQ_SS"]
        ax.bar(sim_taa - 2, diff_dawn, width=4, color='green', alpha=0.6, label="Dawn - SS")
        ax.bar(sim_taa + 2, diff_dusk, width=4, color='magenta', alpha=0.6, label="Dusk - SS")
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_title("Difference relative to Sub-Solar")
        ax.set_ylabel("Delta Column Density")
        ax.set_xlabel("True Anomaly Angle (deg)")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    elif PLOT_MODE == "MILILLO":
        # 既存のNorth/Eq/Southプロット (省略)
        pass