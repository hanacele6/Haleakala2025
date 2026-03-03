import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import pandas as pd
import matplotlib.ticker as ticker

# =========================================================
# 1. 物理定数と正規化因子
# =========================================================
RM_m = 2.440e6  # 水星の半径 [m]
RM_cm = RM_m * 100.0
CM_PER_M = 100.0
CM2_PER_M2 = CM_PER_M * CM_PER_M


# --- Milillo et al. (2021) 比較用の領域面積計算 ---
# 幾何学的定義: ディスク上の y = 0.3 Rm, y = -0.3 Rm で切った面積
# 扇形の面積公式などを利用して算出
def calculate_segment_area(radius_cm, h_ratio):
    """
    円の分割領域（弓形）の面積を計算する
    h_ratio: 中心からの距離 (0.3など)
    """
    theta = 2.0 * np.arccos(h_ratio)  # 中心角 (ラジアン)
    area_segment = (radius_cm ** 2 / 2.0) * (theta - np.sin(theta))
    return area_segment


# 全円盤面積
AREA_FULL_DISK = np.pi * RM_cm ** 2

# 北側・南側の面積 ( > 0.3 Rm, < -0.3 Rm )
# thetaに対応する扇形から三角形を引いたもの = 弓形
# ただし、y > 0.3 Rm は「弓形」の面積そのもの
AREA_NORTH = calculate_segment_area(RM_cm, 0.3)
AREA_SOUTH = AREA_NORTH

# 赤道領域の面積 ( -0.3 Rm < y < 0.3 Rm )
# 全円 - (北 + 南)
AREA_EQUATOR = AREA_FULL_DISK - (AREA_NORTH + AREA_SOUTH)

# 従来の正規化面積 (参考用)
NORMALIZATION_AREA_CM2 = 3.7408e17
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 / 2.0

# --- 解析モデル用定数 ---
G = 6.67384e-5
MS = 1.9884e30
MM = 3.3e23
AU = 1.495978707e13
E = 0.2056
A_AU = 0.3871
L_AU = 0.37078

# 解析モデルパラメータ
TAU0 = 169200.0
PHI0 = 4.6e7
D_PARAM = 4.6e7
ROT_ANGLE_ACCUM = 90

print(f"Milillo比較用 面積計算:")
print(f"  North (>0.3Rm)  : {AREA_NORTH:.4e} cm^2")
print(f"  Equator (+-0.3Rm): {AREA_EQUATOR:.4e} cm^2")
print(f"  South (<-0.3Rm) : {AREA_SOUTH:.4e} cm^2")

# =========================================================
# 2. ユーザー設定
# =========================================================

# ★★★ グリッド設定 (シミュレーション時と同じにする)
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★★★ シミュレーション結果のディレクトリ設定
# --- 1つ目のシミュレーション (メイン) ---
output_dir_1 = r"./SimulationResult_202602/ParabolicHop_72x36_NoEq_DT100_0211_0.4Denabled_2.7_LowestQ_Bounce525K"
label_sim_1 = "Diff Model (U=2.7, Ea=0.4)"

# --- 2つ目のシミュレーション (比較用) ---
ENABLE_SECOND_SIM = False  # 必要に応じてTrueに
output_dir_2 = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0128_0.4Denabled_2.7_LowestQ_test"
label_sim_2 = "Old Model"

# ★★★ プロットモード選択
# "ALL": 従来のDawn/Dusk比較
# "MILILLO": Milillo論文 図3形式 (北/赤道/南) のプロット
PLOT_MODE = "ALL"  # <--- ここを変更しました

# ★★★ CSVプロット (Mililloモードでは無効化または別扱い推奨)
SHOW_CSV_OVERLAY = False
CSV_SETTINGS = [
    {"path": r"./dawn.csv", "label": "Obs: Dawn", "color": "green", "marker": "x", "type": "DAWN"},
    {"path": r"./dusk.csv", "label": "Obs: Dusk", "color": "magenta", "marker": "+", "type": "DUSK"}
]

COMMON_Y_LABEL = "Column Density [atoms/cm²]"

# --- 3. グリッド計算準備 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3

# インデックス計算
mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2
mid_index_z = (GRID_RESOLUTION - 1) // 2  # 北極南極方向 (Z軸と仮定)

# Milillo領域のインデックス境界を計算
# Z軸 (Axis 2) が南北方向である前提
# グリッド座標系: 中心が0, 単位はセル
# 0.3 Rm をセル数に換算
rm_in_cells = RM_m / cell_size_m
boundary_cells = 0.3 * rm_in_cells

idx_north_start = int(mid_index_z + boundary_cells)  # Z > +0.3 Rm
idx_south_end = int(mid_index_z - boundary_cells)  # Z < -0.3 Rm

print(f"Grid Index Info:")
print(f"  Mid Index (Center): {mid_index_z}")
print(f"  0.3 Rm in cells   : {boundary_cells:.2f}")
print(f"  North Start Index : {idx_north_start}")
print(f"  South End Index   : {idx_south_end}")


# --- 4. 解析モデル計算関数群 (省略せずそのまま利用) ---
def get_sun_distance(taa_deg):
    return L_AU / (1.0 + E * np.cos(np.deg2rad(taa_deg)))


def get_orbital_angular_velocity(r_au):
    return np.sqrt((G / AU ** 3) * MS * L_AU) / r_au ** 2 * np.rad2deg(1)


def get_relative_angular_velocity(taa_deg):
    t_orbit = 2.0 * np.pi * np.sqrt(A_AU ** 3 / ((G / AU ** 3) * (MS + MM)))
    t_rot = t_orbit * (2.0 / 3.0)
    rot_dot = 360.0 / t_rot
    r = get_sun_distance(taa_deg)
    taa_dot = get_orbital_angular_velocity(r)
    return rot_dot - taa_dot


# --- 5. データ処理関数 (Milillo対応版) ---
def process_simulation_data(target_dir, mode):
    try:
        all_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')])
        if not all_files:
            print(f"警告: ディレクトリ '{target_dir}' に有効な .npy ファイルがありません。")
            return None, None
    except FileNotFoundError:
        print(f"警告: ディレクトリ '{target_dir}' が見つかりません。")
        return None, None

    print(f"処理中: {target_dir} (Mode: {mode})")

    sim_results_taa = []

    # モードに応じた結果格納辞書の初期化
    if mode == "MILILLO":
        results_dict = {"NORTH": [], "EQUATOR": [], "SOUTH": []}
    elif mode == "ALL":
        results_dict = {"DAWN": [], "DUSK": []}
    else:
        results_dict = {"SINGLE": []}

    for filename in tqdm(all_files):
        try:
            taa = int(filename.split('_taa')[-1].split('.')[0])
        except (ValueError, IndexError):
            continue

        filepath = os.path.join(target_dir, filename)
        # density_grid_m3 形状: [X(Sun-Planet), Y(Dawn-Dusk), Z(North-South)] と仮定
        density_grid_m3 = np.load(filepath)

        # 昼側のみ切り出し & 総原子数変換
        # X軸: mid_index_x より太陽側
        dayside_grid = density_grid_m3[mid_index_x:, :, :]
        atoms_grid = dayside_grid * cell_volume_m3
        atoms_grid[0, :, :] *= 0.5  # Terminator面補正

        # --- モード別集計 ---
        if mode == "MILILLO":
            # Z軸 (Axis 2) でスライスして集計

            # North: Z > 0.3 Rm
            # atoms_gridの形状は [X_half, Y_full, Z_full] -> Axis 2 is Z
            sum_north = np.sum(atoms_grid[:, :, idx_north_start:])
            dens_north = sum_north / AREA_NORTH
            results_dict["NORTH"].append(dens_north)

            # Equator: -0.3 Rm <= Z <= 0.3 Rm
            # Pythonのスライスは start:end (endは含まない) なので +1 調整が必要な場合もあるが
            # インデックス計算で整数化しているため近似とする
            sum_equator = np.sum(atoms_grid[:, :, idx_south_end:idx_north_start])
            dens_equator = sum_equator / AREA_EQUATOR
            results_dict["EQUATOR"].append(dens_equator)

            # South: Z < -0.3 Rm
            sum_south = np.sum(atoms_grid[:, :, :idx_south_end])
            dens_south = sum_south / AREA_SOUTH
            results_dict["SOUTH"].append(dens_south)

        elif mode == "ALL":
            # 従来のDawn/Dusk分割 (Y軸: Axis 1)
            sum_mid = np.sum(atoms_grid[:, mid_index_y, :])

            sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
            total_atoms_dawn = sum_dawn + (0.5 * sum_mid)
            dens_dawn = total_atoms_dawn / NORMALIZATION_AREA_HALF_CM2
            results_dict["DAWN"].append(dens_dawn)

            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            total_atoms_dusk = sum_dusk + (0.5 * sum_mid)
            dens_dusk = total_atoms_dusk / NORMALIZATION_AREA_HALF_CM2
            results_dict["DUSK"].append(dens_dusk)

        else:
            # その他の単一モード（簡易実装）
            total_atoms = np.sum(atoms_grid)
            dens = total_atoms / NORMALIZATION_AREA_CM2
            results_dict["SINGLE"].append(dens)

        sim_results_taa.append(taa)

    # ソート処理
    sim_results_taa = np.array(sim_results_taa)
    sorted_idx = np.argsort(sim_results_taa)
    sim_results_taa = sim_results_taa[sorted_idx]

    final_dict = {}
    for key, val_list in results_dict.items():
        final_dict[key] = np.array(val_list)[sorted_idx]

    return sim_results_taa, final_dict


# --- 6. メイン処理とプロット ---

# シミュレーションデータの読み込み
sim_taa_1, sim_data_1 = process_simulation_data(output_dir_1, PLOT_MODE)

sim_taa_2 = None
sim_data_2 = None
if ENABLE_SECOND_SIM:
    sim_taa_2, sim_data_2 = process_simulation_data(output_dir_2, PLOT_MODE)

# --- プロット描画 ---
if sim_taa_1 is not None:

    if PLOT_MODE == "MILILLO":
        regions = ["NORTH", "EQUATOR", "SOUTH"]
        colors = ["blue", "green", "red"]
        titles = ["North (> 0.3 Rm)", "Equator (+/- 0.3 Rm)", "South (< -0.3 Rm)"]

        # ★ Milillo論文から抽出したCSVファイルのパス指定
        obs_csv_paths = {
            "NORTH": "milillo_north.csv",
            "EQUATOR": "milillo_equator.csv",
            "SOUTH": "milillo_south.csv"
        }

        # 全プロットのY軸を合わせるための最大値取得 (必要に応じて使用)
        max_y = 0
        all_vals = []
        if sim_data_1:
            for r in regions: all_vals.extend(sim_data_1[r])
        if ENABLE_SECOND_SIM and sim_data_2:
            for r in regions: all_vals.extend(sim_data_2[r])
        if all_vals:
            max_y = np.max(all_vals) * 1.2  # 少し余裕を持たせる

        # 領域ごとに別々のウィンドウ(Figure)を生成してループ
        for i, region in enumerate(regions):
            fig, ax = plt.subplots(figsize=(8, 5))

            # 1. Sim 1のプロット
            if sim_data_1:
                ax.plot(sim_taa_1, sim_data_1[region],
                        color=colors[i], marker="o", linestyle="-",
                        label=f"Sim: {label_sim_1}")

            # 2. Sim 2のプロット
            if ENABLE_SECOND_SIM and sim_data_2:
                ax.plot(sim_taa_2, sim_data_2[region],
                        color="gray", marker="s", linestyle="--", alpha=0.7,
                        label=f"Sim: {label_sim_2}")

            # 3. Milillo観測データ (CSV) の重ねプロット
            csv_path = obs_csv_paths[region]
            if os.path.exists(csv_path):
                try:
                    # CSV読み込み (1列目:TAA, 2列目:Density と仮定)
                    df_obs = pd.read_csv(csv_path, header=None)  # ヘッダーがある場合は header=0 に変更
                    obs_taa = df_obs.iloc[:, 0].values
                    obs_dens = df_obs.iloc[:, 1].values

                    # Mililloのグラフは単位が 10^11 atoms/cm2 なので、
                    # シミュレーションの単位に合わせるための倍率が必要な場合があります。
                    # もしCSVの値が「1.5」などで、Simが「1.5e11」なら、以下のように変換します。
                    # obs_dens = obs_dens * 1e11  <-- 必要に応じてコメントアウトを解除

                    ax.scatter(obs_taa, obs_dens, color='black', marker='*', s=100, zorder=5,
                               label="Milillo et al. (2021) Obs")
                except Exception as e:
                    print(f"CSV読み込みエラー ({csv_path}): {e}")
            else:
                print(f"※ 観測データ {csv_path} が見つかりません。シミュレーションのみプロットします。")

            # 軸と見た目の設定
            ax.set_xlabel('True Anomaly Angle (deg)', fontsize=14)
            ax.set_ylabel(COMMON_Y_LABEL, fontsize=14)
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))

            # Y軸のスケールを統一したい場合は下のコメントを外す
            # ax.set_ylim(0, max_y)

            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_title(f"Milillo Comparison: {titles[i]}", fontsize=16)

            # 遠日点・近日点のガイドライン
            ax.axvline(x=180, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=360, color='gray', linestyle=':', alpha=0.5)

            ax.legend(loc="upper left")  # 凡例の位置はお好みで調整
            plt.tight_layout()

            # ウィンドウを個別に表示（JupyterNotebook等ではセルに連続して表示されます）
            plt.show()

    else:
        # 従来のALLモードなどのプロット (既存コードのまま)
        fig, ax1 = plt.subplots(figsize=(10, 7))

        # Sim 1
        if "DAWN" in sim_data_1:
            ax1.plot(sim_taa_1, sim_data_1["DAWN"], color="blue", marker="^", label="Dawn")
            ax1.plot(sim_taa_1, sim_data_1["DUSK"], color="red", marker="v", label="Dusk")
        elif "SINGLE" in sim_data_1:
            ax1.plot(sim_taa_1, sim_data_1["SINGLE"], color="black", label="Total")

        ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=16)
        ax1.set_ylabel(COMMON_Y_LABEL, fontsize=16)
        ax1.grid(True)
        ax1.legend()
        plt.show()

else:
    print("データ処理に失敗しました。")