# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import sys
import pandas as pd
import matplotlib.ticker as ticker
import re

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

# 水星の1年(公転周期)の時間 [hours]
MERCURY_YEAR_HOURS = 87.969 * 24 

print(f"正規化面積 (全体): {NORMALIZATION_AREA_CM2:.4e} cm^2")
print(f"正規化面積 (1/2): {NORMALIZATION_AREA_HALF_CM2:.4e} cm^2")
print(f"正規化面積 (1/4): {NORMALIZATION_AREA_QUARTER_CM2:.4e} cm^2")

# --- 2. ユーザー設定 ---

# ★★★ グリッド設定
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★★★ シミュレーション結果のディレクトリ
output_dir = r"./SimulationResult_202605/ParabolicHop_72x36_NoEq_DT100_0512_Multi_BD0.4_UG-0.2_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_V18"

# ★★★ フィッティング評価設定 ★★★
EVALUATE_FIT = True  # 最小二乗誤差(RMSE等)を計算するかどうか
BIN_SIZE_DEG = 30.0  # TAAのビン幅 [度] (例: 10度ごとに平均化)
SHOW_ERROR_BARS = True  # グラフに誤差棒を表示するか (True/False)
# 除外するTAA領域のリスト [(start1, end1), (start2, end2), ...]
EXCLUDE_TAA_RANGES = []

# ★★★ プロット対象のシミュレーション年 (スピンアップ対応) ★★★
# 1: 1年目, 2: 2年目, 3: 3年目(最終定常状態)
# "ALL" を指定した場合は全データをそのままプロットします（年ごとに色分けされます）
TARGET_YEAR = "ALL"

# ★★★ プロットモード選択 (シミュレーション側)
# 選択肢: "ALL", "DAYSIDE_TOTAL", "DAWN", "DUSK", "DAWN_OUTER", "DUSK_OUTER", "CSV_ONLY"
PLOT_MODE = "ALL"

# ★★★ CSVプロット選択モード (観測データ側)
# "DAWN" : DawnのCSVのみ表示, "DUSK" : DuskのCSVのみ表示, "BOTH" : 両方表示
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
        "path": r"C:\Users\hanac\univ\Mercury/DAWN.csv",
        "label": "Observation: Dawn",
        "color": "green",
        "marker": "x",
        "type": "DAWN"  # 判定用タグ
    },
    {
        "path": r"C:\Users\hanac\univ\Mercury/DUSK.csv",
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
def process_simulation_data(target_dir, mode, target_year):
    try:
        all_files = [f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')]
        if not all_files:
            print(f"エラー: ディレクトリ '{target_dir}' に有効な .npy ファイルがありません。")
            return None, None
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{target_dir}' が見つかりません。")
        return None, None

    # 年数によるフィルタリング
    filtered_files = []
    for f in all_files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', f)
        if match:
            time_h = int(match.group(1))
            taa = int(match.group(2))
            
            # time_h を1年の時間で割り捨て( // )し、1を足すことで年数を算出します
            file_year = int(time_h // MERCURY_YEAR_HOURS) + 1
                
            if target_year != "ALL" and file_year != target_year:
                continue
                
            filtered_files.append((f, time_h, taa))
            
    if not filtered_files:
        return None, None

    # 時間順にソート
    filtered_files.sort(key=lambda x: x[1])

    sim_results_taa = []
    results_dict = {"DAWN": [], "DUSK": []}
    single_result_density = []

    for filename, time_h, taa in tqdm(filtered_files, desc=f"Processing Year {target_year}"):
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
    # TAAでソート (折れ線グラフが往復しないようにする)
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

sim_results = {}
all_years = []

# CSVのみの場合はシミュレーション処理をスキップ
if PLOT_MODE != "CSV_ONLY":
    try:
        # ディレクトリ内のデータを調べて、存在する全シミュレーション年を取得
        for f in os.listdir(output_dir):
            match = re.search(r'_t(\d+)_taa\d+\.npy$', f)
            if match:
                time_h = int(match.group(1))
                file_year = int(time_h // MERCURY_YEAR_HOURS) + 1
                if file_year not in all_years:
                    all_years.append(file_year)
        all_years.sort()
    except FileNotFoundError:
        pass

    if TARGET_YEAR == "ALL":
        target_years_to_process = all_years
    else:
        target_years_to_process = [TARGET_YEAR]

    # 年ごとにデータを取得
    for y in target_years_to_process:
        taa, data = process_simulation_data(output_dir, PLOT_MODE, y)
        if taa is not None and len(taa) > 0:
            sim_results[y] = {"taa": taa, "data": data}

    # =======================================================
    # 前年比の計算と出力 (TARGET_YEAR == "ALL" の場合のみ)
    # =======================================================
    if TARGET_YEAR == "ALL" and len(sim_results) > 1:
        print("\n" + "="*50)
        print(" 前年比 (各年における平均カラム密度の比較)")
        print("="*50)
        sorted_years = sorted(list(sim_results.keys()))
        for i in range(1, len(sorted_years)):
            y_prev = sorted_years[i-1]
            y_curr = sorted_years[i]
            
            data_prev = sim_results[y_prev]["data"]
            data_curr = sim_results[y_curr]["data"]
            
            print(f"--- Year {y_curr} / Year {y_prev} ---")
            if PLOT_MODE == "ALL":
                for key in ["DAWN", "DUSK"]:
                    if len(data_prev[key]) > 0 and len(data_curr[key]) > 0:
                        mean_prev = np.mean(data_prev[key])
                        mean_curr = np.mean(data_curr[key])
                        ratio = mean_curr / mean_prev if mean_prev > 0 else 0
                        diff_percent = (ratio - 1.0) * 100
                        print(f"  {key:4s} : {ratio:.4f} ( {diff_percent:+.2f} % )")
            else:
                if len(data_prev) > 0 and len(data_curr) > 0:
                    mean_prev = np.mean(data_prev)
                    mean_curr = np.mean(data_curr)
                    ratio = mean_curr / mean_prev if mean_prev > 0 else 0
                    diff_percent = (ratio - 1.0) * 100
                    print(f"  {PLOT_MODE} : {ratio:.4f} ( {diff_percent:+.2f} % )")
        print("="*50 + "\n")

    # 既存の評価用・同期用コードのためにフラットな sim_taa, sim_data を作成
    if len(sim_results) > 0:
        sorted_years = sorted(list(sim_results.keys()))
        sim_taa_flat = np.concatenate([sim_results[y]["taa"] for y in sorted_years])
        sort_idx = np.argsort(sim_taa_flat)
        sim_taa = sim_taa_flat[sort_idx]
        
        if PLOT_MODE == "ALL":
            sim_data = {"DAWN": [], "DUSK": []}
            for key in ["DAWN", "DUSK"]:
                data_flat = np.concatenate([sim_results[y]["data"][key] for y in sorted_years])
                sim_data[key] = data_flat[sort_idx]
        else:
            data_flat = np.concatenate([sim_results[y]["data"] for y in sorted_years])
            sim_data = data_flat[sort_idx]
    else:
        sim_taa, sim_data = None, None

else:
    sim_taa, sim_data = None, None


# シミュレーションデータがあるか、CSVのみモードの場合はプロットを実行
if sim_taa is not None or PLOT_MODE == "CSV_ONLY":
    # =======================================================
    # メインウィンドウ (生データ用)
    # =======================================================
    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
    ax1.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')  # 左軸 (共通ラベル)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))
    
    year_title = f" (Year {TARGET_YEAR})" if TARGET_YEAR != "ALL" else " (All Years)"
    plt.title(f"Column Density vs TAA{year_title} (Raw Data)", fontsize=16)

    # --- シミュレーションデータのプロット (メインウィンドウ) ---
    y1_max_data = 0
    if PLOT_MODE != "CSV_ONLY" and len(sim_results) > 0:
        sorted_years = sorted(list(sim_results.keys()))
        total_y = len(sorted_years)
        
        for idx, y in enumerate(sorted_years):
            s_taa = sim_results[y]["taa"]
            s_data = sim_results[y]["data"]
            
            # 年ごとに色を濃くしていく (グラデーション)
            intensity = 0.4 + 0.6 * (idx / max(1, total_y - 1))
            
            if PLOT_MODE == "ALL":
                color_dawn = plt.get_cmap('Blues')(intensity) if total_y > 1 else 'blue'
                color_dusk = plt.get_cmap('Reds')(intensity) if total_y > 1 else 'red'
                
                styles = {
                    "DAWN": {"color": color_dawn, "marker": "^", "label": f"Sim: Dawn (Yr {y})"},
                    "DUSK": {"color": color_dusk, "marker": "v", "label": f"Sim: Dusk (Yr {y})"}
                }
                for key, val_array in s_data.items():
                    st = styles.get(key)
                    ax1.plot(s_taa, val_array,
                             color=st["color"], label=st["label"],
                             marker=st["marker"], markersize=6, alpha=0.8, linestyle='None') 
                    if len(val_array) > 0:
                        y1_max_data = max(y1_max_data, np.max(val_array))
            else:
                color_sim = plt.get_cmap('Greens')(intensity) if total_y > 1 else 'blue'
                ax1.plot(s_taa, s_data, label=f'Sim: {PLOT_MODE} (Yr {y})', color=color_sim, alpha=0.8, marker='o', linestyle='None')
                if len(s_data) > 0:
                    y1_max_data = max(y1_max_data, np.max(s_data))

    # --- CSV観測データ ---
    y2_max_data = 0
    has_csv_plot = False

    # =======================================================
    # ビン化グラフ用の第2ウィンドウ準備 (EVALUATE_FIT=Trueの時のみ後で作成)
    fig_binned = None
    ax_binned = None
    # =======================================================

    # CSV_ONLYモードの場合は SHOW_CSV_OVERLAY の設定に関わらずCSVを描画する
    if SHOW_CSV_OVERLAY or PLOT_MODE == "CSV_ONLY":
        
        # CSV_ONLYの場合は常にax1を使用する
        if PLOT_MODE == "CSV_ONLY" or CSV_USE_SHARED_Y_AXIS:
            target_ax = ax1
        else:
            target_ax = ax1.twinx()
            target_ax.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')  # 右軸 (共通ラベル)
            target_ax.tick_params(axis='y', labelcolor='black')

        for csv_setting in CSV_SETTINGS:
            csv_type = csv_setting.get("type", "UNKNOWN")
            if CSV_PLOT_SELECTION != "BOTH":
                if csv_type != CSV_PLOT_SELECTION:
                    continue 

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
                        csv_taa = df.iloc[:, 2].values
                        csv_density = df.iloc[:, 3].values * 1e11
                        
                        if df.shape[1] >= 5:
                            csv_error = df.iloc[:, 4].values * 1e10
                        else:
                            csv_error = None 

                        if SHOW_ERROR_BARS and csv_error is not None:
                            target_ax.errorbar(csv_taa, csv_density, yerr=csv_error, 
                                               label=csv_label + " (Raw)", color=csv_color, ecolor='black', fmt=csv_marker,
                                               markersize=6, capsize=2, elinewidth=1.0, 
                                               alpha=1.0, zorder=2, linestyle='None')
                        else:
                            target_ax.scatter(csv_taa, csv_density, label=csv_label + " (Raw)", 
                                              color=csv_color, marker=csv_marker,
                                              s=40, zorder=2, alpha=1.0) 

                        if len(csv_density) > 0:
                            y2_max_data = max(y2_max_data, np.max(csv_density))
                        has_csv_plot = True

                        # =======================================================
                        # 2. ビン化計算とモデルの評価（EVALUATE_FIT == True の時のみ）
                        # =======================================================
                        if EVALUATE_FIT and PLOT_MODE == "ALL" and sim_taa is not None:
                            if csv_type in sim_data:
                                model_val = sim_data[csv_type]
                                
                                bin_edges = np.arange(0, 360 + BIN_SIZE_DEG, BIN_SIZE_DEG)
                                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                                
                                binned_obs = []
                                binned_err = []
                                binned_mod = []
                                valid_bin_centers = []

                                for i in range(len(bin_centers)):
                                    b_start = bin_edges[i]
                                    b_end = bin_edges[i+1]
                                    b_center = bin_centers[i]
                                    
                                    is_excluded = False
                                    for (t_start, t_end) in EXCLUDE_TAA_RANGES:
                                        if (t_start <= b_center <= t_end):
                                            is_excluded = True
                                            break
                                    if is_excluded: continue

                                    mask = (csv_taa >= b_start) & (csv_taa < b_end)
                                    if np.any(mask):
                                        obs_mean = np.mean(csv_density[mask])
                                        
                                        if csv_error is not None:
                                            err_mean = np.mean(csv_error[mask])
                                        else:
                                            err_mean = obs_mean * 0.10

                                        # モデル側もビン内の区間平均を取得
                                        dense_taa = np.linspace(b_start, b_end, 20)
                                        dense_mod = np.interp(dense_taa, sim_taa, model_val)
                                        mod_mean = np.mean(dense_mod)
                                        
                                        binned_obs.append(obs_mean)
                                        binned_err.append(err_mean)
                                        binned_mod.append(mod_mean)
                                        valid_bin_centers.append(b_center)

                                binned_obs = np.array(binned_obs)
                                binned_err = np.array(binned_err)
                                binned_mod = np.array(binned_mod)
                                valid_bin_centers = np.array(valid_bin_centers)

                                # =======================================================
                                # 3. 第2ウィンドウ(Figure)にビン化データをプロット
                                # =======================================================
                                if len(binned_obs) > 0:
                                    if fig_binned is None:
                                        fig_binned, ax_binned = plt.subplots(figsize=(10, 7))
                                        ax_binned.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
                                        ax_binned.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')
                                        ax_binned.tick_params(axis='both', which='major', labelsize=14)
                                        ax_binned.yaxis.get_offset_text().set_fontsize(14)
                                        ax_binned.set_xlim(0, 360)
                                        ax_binned.set_xticks(np.arange(0, 361, 60))
                                        ax_binned.set_title(f"Binned Evaluation: Obs vs Model{year_title} (Bin Size: {BIN_SIZE_DEG}°)", fontsize=16)

                                    if SHOW_ERROR_BARS:
                                        ax_binned.errorbar(valid_bin_centers, binned_obs, 
                                                           yerr=binned_err, xerr=BIN_SIZE_DEG/2.0, 
                                                           label=f"Obs Binned: {csv_type}", 
                                                           color='black', ecolor='black', fmt='o',
                                                           markersize=5, capsize=3, elinewidth=1.0,
                                                           alpha=1.0, zorder=4, linestyle='None')
                                        ax_binned.scatter(valid_bin_centers, binned_obs, 
                                                          color=csv_color, marker='o', s=50, 
                                                          edgecolors='black', linewidths=0.5, zorder=5)
                                    else:
                                        ax_binned.scatter(valid_bin_centers, binned_obs, label=f"Obs: {csv_type}", 
                                                          color=csv_color, marker='o', s=50, 
                                                          edgecolors='black', linewidths=0.5, zorder=5)

                                    # ビン化されたモデル（階段状グラフ: drawstyle='steps-mid'）
                                    ax_binned.plot(valid_bin_centers, binned_mod, label=f"Model Binned: {csv_type}", 
                                                   color=csv_color, zorder=3, linestyle='-', linewidth=1.5, drawstyle='steps-mid')

                                # =======================================================
                                # 4. 統計指標（換算χ二乗など）の出力
                                # =======================================================
                                if len(binned_obs) > 0:
                                    residuals = binned_mod - binned_obs
                                    sse = np.sum(residuals**2)
                                    rmse = np.sqrt(np.mean(residuals**2))
                                    mean_obs = np.mean(binned_obs)
                                    nrmsd = (rmse / mean_obs) * 100 if mean_obs > 0 else 0

                                    dof = len(binned_obs) - 0
                                    sigma = np.where(binned_err == 0, 1e-10, binned_err)

                                    if dof > 0:
                                        chi_square = np.sum((residuals / sigma)**2)
                                        reduced_chi_square = chi_square / dof
                                    else:
                                        reduced_chi_square = float('inf')

                                    print(f"--- Binned Fit Evaluation: {csv_label} ---")
                                    print(f"  Valid Bins   : {len(binned_obs)} (Bin Size: {BIN_SIZE_DEG} deg)")
                                    print(f"  RMSE         : {rmse:.4e} [atoms/cm²]")
                                    print(f"  NRMSD        : {nrmsd:.2f} % (対平均値)")
                                    if csv_error is not None:
                                        print(f"  Reduced χ²   : {reduced_chi_square:.4f}")
                                    else:
                                        print(f"  Reduced χ²   : {reduced_chi_square:.4f} (※誤差10%仮定)")
                                else:
                                    print(f"--- Binned Fit Evaluation: {csv_label} ---")
                                    print("  エラー: 有効なTAA領域にデータがありません。")

                except Exception as e:
                    print(f"CSV error: {e}")

        # --- 軸の同期処理 (シミュレーションが存在し、かつ右軸がある場合のみ) ---
        if PLOT_MODE != "CSV_ONLY" and not CSV_USE_SHARED_Y_AXIS and has_csv_plot:

            def get_align_params(taa_arr, val_arr):
                if len(taa_arr) == 0: return 0, 1
                val_max = np.max(val_arr)
                diff = np.abs(np.mod(taa_arr, 360.0))
                diff = np.minimum(diff, 360.0 - diff)
                idx_peri = np.argmin(diff)
                val_peri = val_arr[idx_peri]
                return val_peri, val_max

            if isinstance(sim_data, dict):
                vals_list = list(sim_data.values())
                sim_m = np.max([np.max(v) for v in vals_list])
                ref_key = list(sim_data.keys())[0]
                sim_p, _ = get_align_params(sim_taa, sim_data[ref_key])
            else:
                sim_p, sim_m = get_align_params(sim_taa, sim_data)

            obs_vals_list = []
            obs_taas_list = []
            for coll in target_ax.collections:
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    obs_taas_list.append(offsets[:, 0])
                    obs_vals_list.append(offsets[:, 1])

            if obs_vals_list:
                obs_all_taa = np.concatenate(obs_taas_list)
                obs_all_val = np.concatenate(obs_vals_list)
                obs_p, obs_m = get_align_params(obs_all_taa, obs_all_val)
            else:
                obs_p, obs_m = 0, 1

            if sim_p > 0:
                ratio = obs_p / sim_p
            else:
                ratio = 1.0

            print(f"Align Ratio (Obs/Sim at Perihelion): {ratio:.4f}")

            required_sim_top = max(sim_m, obs_m / ratio)
            final_sim_top = required_sim_top * 1.1
            final_obs_top = final_sim_top * ratio

            ax1.set_ylim(0, final_sim_top)
            target_ax.set_ylim(0, final_obs_top)
            target_ax.grid(False)

        # CSVのみモードの場合のY軸自動調整
        if PLOT_MODE == "CSV_ONLY" and y2_max_data > 0:
            ax1.set_ylim(0, y2_max_data * 1.1)

    # =======================================================
    # 最終レイアウト調整と表示
    # =======================================================
    
    # 1. メインウィンドウの調整
    ax1.set_ylim(bottom=0)
    if 'target_ax' in locals() and target_ax != ax1:
        target_ax.set_ylim(bottom=0)

    if SHOW_LEGEND:
        if PLOT_MODE != "CSV_ONLY" and not CSV_USE_SHARED_Y_AXIS and has_csv_plot:
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = target_ax.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)
        else:
            ax1.legend(loc='upper left', fontsize=12)
            
    fig.tight_layout()

    # 2. 第2ウィンドウ（ビン化グラフ）の調整
    if fig_binned is not None:
        ax_binned.set_ylim(bottom=0)
        if SHOW_LEGEND:
            ax_binned.legend(loc='upper left', fontsize=12)
        fig_binned.tight_layout()

    # 両方のウィンドウを表示
    plt.show()

else:
    print("データ処理に失敗したため終了します。")