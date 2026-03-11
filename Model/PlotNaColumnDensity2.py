import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import pandas as pd
import matplotlib.ticker as ticker

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
#output_dir = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0128_0.4Denabled_2.7_LowestQ_test"
#output_dir = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_1224_s1"
#output_dir = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0201_0.4Denabled_1.85_LowestQ_test"
#output_dir = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0109_0.4Denabled_2.7_HalfQ"
#output_dir = r"./SimulationResult_202602/HeteroSurf_72x36_NoEq_DT100_UDist_Refill1.0e-5"
#output_dir = r"./SimulationResult_202602/ParabolicHop_72x36_NoEq_DT100_0211_0.4Denabled_2.7_LowestQ_Bounce525K"
output_dir = r"./SimulationResult_202602/ParabolicHop_72x36_NoEq_DT100_0306_0.65Denabled_2.7_LowestQ_Bounce525K_A2.0_LongLT"


# ★★★ プロットモード選択 (シミュレーション側)
# 選択肢: "ALL", "DAYSIDE_TOTAL", "DAWN", "DUSK", "DAWN_OUTER", "DUSK_OUTER"
PLOT_MODE = "ALL"
#PLOT_MODE = "DAYSIDE_TOTAL"

# ★★★ CSVプロット選択モード (観測データ側)
# "DAWN" : DawnのCSVのみ表示
# "DUSK" : DuskのCSVのみ表示
# "BOTH" : 両方表示
CSV_PLOT_SELECTION = "BOTH"

# ★★★ 軸ラベル名 (共通)
COMMON_Y_LABEL = "Column Density [atoms/cm²]"

# ★★★ 凡例を表示するか (True: 表示 / False: 非表示)
SHOW_LEGEND = True

# ★★★ 比較用CSVファイルの設定
SHOW_CSV_OVERLAY = True  # CSVを重ねて表示するか
CSV_USE_SHARED_Y_AXIS = True  # True: 左軸を共有 / False: 右軸を使用(スケール分離)

# ★★★ 複数CSVの設定リスト ("type"キーで判定します)
CSV_SETTINGS = [
    {
        "path": r"./dawn.csv",
        "label": "Observation: Dawn",
        "color": "green",
        "marker": "x",
        "type": "DAWN"  # 判定用タグ
    },
    {
        "path": r"./dusk.csv",
        "label": "Observation: Dusk",
        "color": "magenta",
        "marker": "+",
        "type": "DUSK"  # 判定用タグ
    }
]

# --- 3. グリッド計算準備 ---
grid_total_width_m = 2 * GRID_MAX_RM * RM_m
cell_size_m = grid_total_width_m / GRID_RESOLUTION
cell_volume_m3 = cell_size_m ** 3
print(f"グリッド解像度: {GRID_RESOLUTION}x{GRID_RESOLUTION}x{GRID_RESOLUTION}")

mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2
quarter_index_offset = mid_index_y // 2
idx_dawn_outer_limit = quarter_index_offset
idx_dusk_outer_start = (GRID_RESOLUTION - 1) - quarter_index_offset


# --- 4. データ処理関数 ---
def process_simulation_data(target_dir, mode):
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

    sim_results_taa = []
    results_dict = {"DAWN": [], "DUSK": []}
    single_result_density = []

    for filename in tqdm(all_files):
        try:
            taa = int(filename.split('_taa')[-1].split('.')[0])
        except (ValueError, IndexError):
            continue

        filepath = os.path.join(target_dir, filename)
        density_grid_m3 = np.load(filepath)

        # 昼側のみ切り出し & 総数変換
        dayside_grid = density_grid_m3[mid_index_x:, :, :]
        atoms_grid = dayside_grid * cell_volume_m3
        atoms_grid[0, :, :] *= 0.5  # Terminator面補正

        # 共通部分和
        sum_mid = 0
        if mode in ["DAWN", "DUSK", "ALL"]:
            sum_mid = np.sum(atoms_grid[:, mid_index_y, :])

        if mode == "ALL":
            # DAWN
            sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
            total_atoms_dawn = sum_dawn + (0.5 * sum_mid)
            dens_dawn = total_atoms_dawn / NORMALIZATION_AREA_HALF_CM2
            results_dict["DAWN"].append(dens_dawn)

            # DUSK
            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            total_atoms_dusk = sum_dusk + (0.5 * sum_mid)
            dens_dusk = total_atoms_dusk / NORMALIZATION_AREA_HALF_CM2
            results_dict["DUSK"].append(dens_dusk)

        else:
            # 単一モード計算
            total_atoms = 0.0
            target_area = 1.0

            if mode == "DAYSIDE_TOTAL":
                total_atoms = np.sum(atoms_grid)
                target_area = NORMALIZATION_AREA_CM2
            elif mode == "DAWN":
                sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
                total_atoms = sum_dawn + (0.5 * sum_mid)
                target_area = NORMALIZATION_AREA_HALF_CM2
            elif mode == "DUSK":
                sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
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
            else:
                print(f"不明なモード: {mode}")
                sys.exit()

            col_density = total_atoms / target_area
            single_result_density.append(col_density)

        sim_results_taa.append(taa)

    sim_results_taa = np.array(sim_results_taa)
    sorted_idx = np.argsort(sim_results_taa)
    sim_results_taa = sim_results_taa[sorted_idx]

    if mode == "ALL":
        final_dict = {}
        for key, val_list in results_dict.items():
            final_dict[key] = np.array(val_list)[sorted_idx]
        return sim_results_taa, final_dict
    else:
        return sim_results_taa, np.array(single_result_density)[sorted_idx]


# --- 5. メイン処理とプロット ---

sim_taa, sim_data = process_simulation_data(output_dir, PLOT_MODE)

if sim_taa is not None:
    fig, ax1 = plt.subplots(figsize=(10, 7))

    # --- 左軸: シミュレーション ---
    y1_max_data = 0

    if PLOT_MODE == "ALL":
        styles = {
            "DAWN": {"color": "blue", "marker": "^", "label": "Simulation: Dawn"},
            "DUSK": {"color": "red", "marker": "v", "label": "Simulation: Dusk"}
        }
        for key, val_array in sim_data.items():
            st = styles.get(key, {"color": "gray", "marker": "o", "label": key})
            ax1.plot(sim_taa, val_array,
                     color=st["color"], label=st["label"],
                     marker=st["marker"], markersize=6, alpha=0.8, linestyle='')
            if len(val_array) > 0:
                y1_max_data = max(y1_max_data, np.max(val_array))
    else:
        color_sim = 'blue'
        ax1.scatter(sim_taa, sim_data, label=f'Simulation: {PLOT_MODE}', color=color_sim, alpha=0.8, s=50)
        if len(sim_data) > 0:
            y1_max_data = np.max(sim_data)

    ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
    ax1.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')  # 左軸 (共通ラベル)
    # ax1.tick_params(axis='y', labelcolor='black')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))

    # --- 右軸: CSV観測データ ---
    y2_max_data = 0
    has_csv_plot = False

    if SHOW_CSV_OVERLAY:
        target_ax = ax1
        if not CSV_USE_SHARED_Y_AXIS:
            target_ax = ax1.twinx()
            target_ax.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')  # 右軸 (共通ラベル)
            target_ax.tick_params(axis='y', labelcolor='black')

        for csv_setting in CSV_SETTINGS:
            # フィルタリング処理
            csv_type = csv_setting.get("type", "UNKNOWN")
            if CSV_PLOT_SELECTION != "BOTH":
                if csv_type != CSV_PLOT_SELECTION:
                    continue  # 選択されていないタイプはスキップ

            csv_path = csv_setting["path"]
            csv_label = csv_setting["label"]
            csv_color = csv_setting.get("color", "green")
            csv_marker = csv_setting.get("marker", "x")

            if os.path.exists(csv_path):
                try:
                    try:
                        df = pd.read_csv(csv_path, encoding='shift_jis')
                    except UnicodeDecodeError:
                        df = pd.read_csv(csv_path, encoding='cp932')

                    if df.shape[1] >= 4:
                        csv_taa = df.iloc[:, 3].values
                        csv_density = df.iloc[:, 4].values

                        # プロット
                        target_ax.scatter(csv_taa, csv_density, label=csv_label, color=csv_color, marker=csv_marker,
                                          s=80, zorder=3, linewidths=1.5)

                        if len(csv_density) > 0:
                            y2_max_data = max(y2_max_data, np.max(csv_density))
                        has_csv_plot = True
                except Exception as e:
                    print(f"CSV error: {e}")

        # --- 軸の同期処理 (右軸がある場合) ---
        if not CSV_USE_SHARED_Y_AXIS and has_csv_plot:

            # === ロジック修正: 近日点比率で合わせつつ、全体が入るようにする ===

            def get_align_params(taa_arr, val_arr):
                """TAA配列と値配列から、近日点付近の値と最大値を取得する"""
                if len(taa_arr) == 0: return 0, 1

                # 1. 最大値 (Peak)
                val_max = np.max(val_arr)

                # 2. 近日点の値 (TAAが0または360に最も近いインデックスの値)
                diff = np.abs(np.mod(taa_arr, 360.0))
                diff = np.minimum(diff, 360.0 - diff)
                idx_peri = np.argmin(diff)
                val_peri = val_arr[idx_peri]

                return val_peri, val_max


            # --- 1. シミュレーション側の基準値取得 ---
            if isinstance(sim_data, dict):
                # "ALL"モードなど辞書の場合、全データの最大値を取得
                vals_list = list(sim_data.values())
                sim_m = np.max([np.max(v) for v in vals_list])
                # 近日点は代表として最初のキーを使用
                ref_key = list(sim_data.keys())[0]
                sim_p, _ = get_align_params(sim_taa, sim_data[ref_key])
            else:
                sim_p, sim_m = get_align_params(sim_taa, sim_data)

            # --- 2. 観測データ(CSV)側の基準値取得 ---
            # プロット済みのデータから再取得（複数CSVがある場合も考慮して結合）
            obs_vals_list = []
            obs_taas_list = []
            for coll in target_ax.collections:
                offsets = coll.get_offsets()  # [[x, y], ...]
                if len(offsets) > 0:
                    obs_taas_list.append(offsets[:, 0])
                    obs_vals_list.append(offsets[:, 1])

            if obs_vals_list:
                obs_all_taa = np.concatenate(obs_taas_list)
                obs_all_val = np.concatenate(obs_vals_list)
                obs_p, obs_m = get_align_params(obs_all_taa, obs_all_val)
            else:
                obs_p, obs_m = 0, 1

            # --- 3. 制限の適用 ---

            # (A) 近日点での比率を計算 (Obs / Sim)
            if sim_p > 0:
                ratio = obs_p / sim_p
            else:
                ratio = 1.0

            print(f"Align Ratio (Obs/Sim at Perihelion): {ratio:.4f}")

            # (B) 必要な軸の上限を計算
            # 左軸(Sim)の上限は、「Simの最大値」と「Obsの最大値をSimスケールに換算した値」のうち
            # **大きい方** に合わせて設定する必要があります。
            # そうしないと、Obs側のピークが画面外にはみ出してしまいます。

            # Obsの最大値をSimのスケールに戻すと -> obs_m / ratio
            required_sim_top = max(sim_m, obs_m / ratio)

            # 余白を10%持たせる
            final_sim_top = required_sim_top * 1.1

            # 右軸(Obs)の上限は、左軸に比率を掛けたもの（これで0と近日点が同期します）
            final_obs_top = final_sim_top * ratio

            # (C) 設定反映 (下限は0固定で全体を表示)
            ax1.set_ylim(0, final_sim_top)
            target_ax.set_ylim(0, final_obs_top)

            # 右軸のグリッドは線がズレて見にくいので消す
            target_ax.grid(False)

            print(f"Axis Scaling Info (0-Aligned + Perihelion Ratio):")
            print(f"  Sim: Peri={sim_p:.2e}, Max={sim_m:.2e} -> Ylim=[0, {final_sim_top:.2e}]")
            print(f"  Obs: Peri={obs_p:.2e}, Max={obs_m:.2e} -> Ylim=[0, {final_obs_top:.2e}]")

        # --- 凡例の統合処理 ---
        if SHOW_LEGEND:
            if not CSV_USE_SHARED_Y_AXIS:
                lines_1, labels_1 = ax1.get_legend_handles_labels()
                lines_2, labels_2 = target_ax.get_legend_handles_labels()
                ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)
            else:
                ax1.legend(loc='upper left', fontsize=12)

    else:
        # CSVなしの場合
        if SHOW_LEGEND:
            ax1.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()

else:
    print("データ処理に失敗したため終了します。")