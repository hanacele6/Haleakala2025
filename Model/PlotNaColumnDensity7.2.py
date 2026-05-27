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

# ★★★ 比較するシミュレーション結果のリスト ★★★
# 複数のモデルを追加・編集できます
MODEL_SETTINGS = [
    {
        "dir": r"./SimulationResult_202605/ParabolicHop_72x36_NoEq_DT100_0509_Multi_BD0.4_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_V18",

        "label": "Model (Mode:1.85 eV)",
        "color_dawn": "blue",    # ALLモード時のDawnの色
        "color_dusk": "red",     # ALLモード時のDuskの色
        "marker_dawn": "^",
        "marker_dusk": "v",
        "color_single": "blue",  # ALL以外の単一モード時の色
        "marker_single": "o",
        "alpha": 0.4,
        "zorder": 1
    },
    # 比較したい別のモデルがある場合は以下のように追加します
     {
         "dir": r"./SimulationResult_202605/ParabolicHop_72x36_NoEq_DT100_0511_Multi_BD0.4_UG+0.2_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)",

         "label": "Model (Mode:2.00 eV)",
         "color_dawn": "cyan",
         "color_dusk": "orange",
         "marker_dawn": "^",
         "marker_dusk": "v",
         "color_single": "cyan",
         "marker_single": "s",
         "alpha": 1.0,
         "zorder": 5
     },
    {
        "dir": r"./SimulationResult_202605/ParabolicHop_72x36_NoEq_DT100_0512_Multi_BD0.4_UG-0.2_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_V18",

        "label": "Model (Mode:1.65 eV)",
        "color_dawn": "purple",    # ALLモード時のDawnの色
        "color_dusk": "brown",     # ALLモード時のDuskの色
        "marker_dawn": "^",
        "marker_dusk": "v",
        "color_single": "blue",  # ALL以外の単一モード時の色
        "marker_single": "D",
        "alpha": 0.4,
        "zorder": 1
    },
]

# ★★★ フィッティング評価設定 ★★★
EVALUATE_FIT = False  # 最小二乗誤差(RMSE等)を計算するかどうか
BIN_SIZE_DEG = 1.0  # TAAのビン幅 [度] (例: 10度ごとに平均化)
SHOW_ERROR_BARS = True  # グラフに誤差棒を表示するか (True/False)
# 除外するTAA領域のリスト [(start1, end1), (start2, end2), ...]
EXCLUDE_TAA_RANGES = []

# ★★★ プロット対象のシミュレーション年 (スピンアップ対応) ★★★
TARGET_YEAR = 3

# ★★★ プロットモード選択 (シミュレーション側)
# 選択肢: "ALL", "DAYSIDE_TOTAL", "DAWN", "DUSK", "DAWN_OUTER", "DUSK_OUTER", "CSV_ONLY"
PLOT_MODE = "DAWN"

# ★★★ CSVプロット選択モード (観測データ側)
# "DAWN" : DawnのCSVのみ表示, "DUSK" : DuskのCSVのみ表示, "BOTH" : 両方表示
CSV_PLOT_SELECTION = "DAWN"

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
        "type": "DAWN"
    },
    {
        "path": r"C:\Users\hanac\univ\Mercury/DUSK.csv",
        "label": "Observation: Dusk",
        "color": "magenta",
        "marker": "+",
        "type": "DUSK"
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

    filtered_files = []
    for f in all_files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', f)
        if match:
            time_h = int(match.group(1))
            taa = int(match.group(2))
            
            file_year = int(time_h // MERCURY_YEAR_HOURS) + 1
                
            if target_year != "ALL" and file_year != target_year:
                continue
                
            filtered_files.append((f, time_h, taa))
            
    if not filtered_files:
        print(f"警告: 指定された年 (Year {target_year}) のデータが見つかりません。")
        return None, None

    filtered_files.sort(key=lambda x: x[1])

    sim_results_taa = []
    results_dict = {"DAWN": [], "DUSK": []}
    single_result_density = []

    for filename, time_h, taa in tqdm(filtered_files, desc=f"Loading {os.path.basename(target_dir)[:15]}..."):
        filepath = os.path.join(target_dir, filename)
        density_grid_m3 = np.load(filepath)

        dayside_grid = density_grid_m3[mid_index_x:, :, :]
        atoms_grid = dayside_grid * cell_volume_m3
        atoms_grid[0, :, :] *= 0.5  # Terminator面補正

        sum_mid = 0
        if mode in ["DAWN", "DUSK", "ALL"]:
            sum_mid = np.sum(atoms_grid[:, mid_index_y, :])

        if mode == "ALL":
            sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
            total_atoms_dawn = sum_dawn + (0.5 * sum_mid)
            dens_dawn = total_atoms_dawn / NORMALIZATION_AREA_HALF_CM2
            results_dict["DAWN"].append(dens_dawn)

            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            total_atoms_dusk = sum_dusk + (0.5 * sum_mid)
            dens_dusk = total_atoms_dusk / NORMALIZATION_AREA_HALF_CM2
            results_dict["DUSK"].append(dens_dusk)

        else:
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

all_models_data = []

# 全モデルのシミュレーションデータを読み込み
if PLOT_MODE != "CSV_ONLY":
    print(f"処理モード: {PLOT_MODE}")
    year_str = f"Year {TARGET_YEAR}" if TARGET_YEAR != "ALL" else "All Years"
    print(f"対象データ: {year_str}")
    
    for mod_set in MODEL_SETTINGS:
        taa, data = process_simulation_data(mod_set["dir"], PLOT_MODE, TARGET_YEAR)
        if taa is not None:
            all_models_data.append({
                "setting": mod_set,
                "taa": taa,
                "data": data
            })

# シミュレーションデータがあるか、CSVのみモードの場合はプロットを実行
if len(all_models_data) > 0 or PLOT_MODE == "CSV_ONLY":
    # =======================================================
    # メインウィンドウ (生データ用)
    # =======================================================
    fig, ax1 = plt.subplots(figsize=(10, 7))

    ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
    ax1.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')  
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))
    
    year_title = f" (Year {TARGET_YEAR})" if TARGET_YEAR != "ALL" else ""

    # --- シミュレーションデータのプロット (メインウィンドウ) ---
    y1_max_data = 0
    if PLOT_MODE != "CSV_ONLY":
        for mod_info in all_models_data:
            st = mod_info["setting"]
            sim_taa = mod_info["taa"]
            sim_data = mod_info["data"]
            
            # ★ 辞書からalphaを取得。設定がない場合はデフォルト0.8とする
            mod_alpha = st.get("alpha", 0.8)
            mod_zorder = st.get("zorder", 3)
            
            if PLOT_MODE == "ALL":
                # DAWNプロット
                if "DAWN" in sim_data and len(sim_data["DAWN"]) > 0:
                    ax1.plot(sim_taa, sim_data["DAWN"],
                             color=st.get("color_dawn", "blue"), 
                             label=f"{st['label']}: Dawn",
                             marker=st.get("marker_dawn", "^"), markersize=6, 
                             alpha=mod_alpha, zorder=mod_zorder, linestyle='None') # ★ 固定値0.8から mod_alpha に変更
                    y1_max_data = max(y1_max_data, np.max(sim_data["DAWN"]))
                # DUSKプロット
                if "DUSK" in sim_data and len(sim_data["DUSK"]) > 0:
                    ax1.plot(sim_taa, sim_data["DUSK"],
                             color=st.get("color_dusk", "red"), 
                             label=f"{st['label']}: Dusk",
                             marker=st.get("marker_dusk", "v"), markersize=6, 
                             alpha=mod_alpha, zorder=mod_zorder, linestyle='None') # ★ ここも mod_alpha に変更
                    y1_max_data = max(y1_max_data, np.max(sim_data["DUSK"]))
            else:
                # ALL以外の単一モード用
                # PLOT_MODE に合わせて色とマーカーを自動選択する
                if "DAWN" in PLOT_MODE:
                    p_color = st.get("color_dawn", "blue")
                    p_marker = st.get("marker_dawn", "^")
                elif "DUSK" in PLOT_MODE:
                    p_color = st.get("color_dusk", "red")
                    p_marker = st.get("marker_dusk", "v")
                else:
                    p_color = st.get("color_single", "blue")
                    p_marker = st.get("marker_single", "o")

                ax1.plot(sim_taa, sim_data, 
                         label=f"{st['label']}: {PLOT_MODE}", 
                         color=p_color, 
                         alpha=mod_alpha, zorder=mod_zorder,
                         marker=p_marker, linestyle='None')
    # --- CSV観測データ ---
    y2_max_data = 0
    has_csv_plot = False
    
    fig_binned = None
    ax_binned = None

    if SHOW_CSV_OVERLAY or PLOT_MODE == "CSV_ONLY":
        if PLOT_MODE == "CSV_ONLY" or CSV_USE_SHARED_Y_AXIS:
            target_ax = ax1
        else:
            target_ax = ax1.twinx()
            target_ax.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')
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
                                               label=csv_label, color=csv_color, ecolor='black', fmt=csv_marker,
                                               markersize=6, capsize=2, elinewidth=1.0, 
                                               alpha=1.0, zorder=2, linestyle='None')
                        else:
                            target_ax.scatter(csv_taa, csv_density, label=csv_label, 
                                              color=csv_color, marker=csv_marker,
                                              s=40, zorder=2, alpha=1.0) 

                        if len(csv_density) > 0:
                            y2_max_data = max(y2_max_data, np.max(csv_density))
                        has_csv_plot = True

                        # =======================================================
                        # 2. ビン化計算とモデルの評価
                        # =======================================================
                        if EVALUATE_FIT and PLOT_MODE == "ALL" and len(all_models_data) > 0:
                            
                            bin_edges = np.arange(0, 360 + BIN_SIZE_DEG, BIN_SIZE_DEG)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                            
                            # 観測データのビン化 (モデルに依存しないので先に1回計算)
                            binned_obs = []
                            binned_err = []
                            valid_bin_centers = []
                            
                            for i in range(len(bin_centers)):
                                b_start, b_end, b_center = bin_edges[i], bin_edges[i+1], bin_centers[i]
                                is_excluded = any(t_start <= b_center <= t_end for t_start, t_end in EXCLUDE_TAA_RANGES)
                                if is_excluded: continue

                                mask = (csv_taa >= b_start) & (csv_taa < b_end)
                                if np.any(mask):
                                    obs_mean = np.mean(csv_density[mask])
                                    if csv_error is not None:
                                        n_data = len(csv_density[mask])
                                        if n_data > 1:
                                            term1_sq = np.sum(csv_error[mask]**2) / n_data
                                            term2_sq = np.var(csv_density[mask], ddof=1)
                                            err_mean = np.sqrt(term1_sq + term2_sq)
                                        elif n_data == 1:
                                            err_mean = csv_error[mask][0]
                                        else:
                                            err_mean = 0.0
                                    else:
                                        err_mean = obs_mean * 0.10
                                        
                                    binned_obs.append(obs_mean)
                                    binned_err.append(err_mean)
                                    valid_bin_centers.append(b_center)

                            binned_obs = np.array(binned_obs)
                            binned_err = np.array(binned_err)
                            valid_bin_centers = np.array(valid_bin_centers)

                            if len(binned_obs) > 0:
                                if fig_binned is None:
                                    fig_binned, ax_binned = plt.subplots(figsize=(10, 7))
                                    ax_binned.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
                                    ax_binned.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')
                                    ax_binned.tick_params(axis='both', which='major', labelsize=14)
                                    ax_binned.yaxis.get_offset_text().set_fontsize(14)
                                    ax_binned.set_xlim(0, 360)
                                    ax_binned.set_xticks(np.arange(0, 361, 60))

                                # 観測プロット (ビン化)
                                if SHOW_ERROR_BARS:
                                    ax_binned.errorbar(valid_bin_centers, binned_obs, 
                                                       yerr=binned_err, xerr=BIN_SIZE_DEG/2.0, 
                                                       label=f"Obs Binned: {csv_type}", 
                                                       color='black', ecolor='black', fmt='o',
                                                       markersize=5, capsize=3, elinewidth=1.0,
                                                       alpha=1.0, zorder=4, linestyle='None')
                                ax_binned.scatter(valid_bin_centers, binned_obs, color=csv_color, marker='o', s=50, edgecolors='black', linewidths=0.5, zorder=5)

                                # 各モデルに対するビン化評価ループ
                                print(f"\n--- Binned Fit Evaluation: {csv_label} ---")
                                print(f"  Valid Bins   : {len(binned_obs)} (Bin Size: {BIN_SIZE_DEG} deg)")
                                
                                for mod_info in all_models_data:
                                    st = mod_info["setting"]
                                    sim_taa = mod_info["taa"]
                                    sim_data = mod_info["data"]
                                    
                                    if csv_type in sim_data:
                                        model_val = sim_data[csv_type]
                                        binned_mod = []
                                        
                                        for i in range(len(valid_bin_centers)):
                                            b_center = valid_bin_centers[i]
                                            b_start = b_center - BIN_SIZE_DEG/2.0
                                            b_end = b_center + BIN_SIZE_DEG/2.0
                                            
                                            dense_taa = np.linspace(b_start, b_end, 20)
                                            dense_mod = np.interp(dense_taa, sim_taa, model_val)
                                            binned_mod.append(np.mean(dense_mod))
                                            
                                        binned_mod = np.array(binned_mod)

                                        # モデルプロット (階段状)
                                        step_x, step_y = [], []
                                        half_bin = BIN_SIZE_DEG / 2.0
                                        for j in range(len(valid_bin_centers)):
                                            vx, vy = valid_bin_centers[j], binned_mod[j]
                                            step_x.extend([vx - half_bin, vx + half_bin])
                                            step_y.extend([vy, vy])
                                            if j < len(valid_bin_centers) - 1:
                                                if not np.isclose(vx + BIN_SIZE_DEG, valid_bin_centers[j+1]):
                                                    step_x.append(np.nan)
                                                    step_y.append(np.nan)

                                        mod_color = st.get("color_dawn") if csv_type == "DAWN" else st.get("color_dusk")
                                        mod_alpha = st.get("alpha", 0.8)
                                        ax_binned.plot(step_x, step_y, label=f"{st['label']}: {csv_type}", 
                                                       color=mod_color, zorder=3, linestyle='-', linewidth=1.5,alpha=mod_alpha)

                                        # 統計出力
                                        residuals = binned_mod - binned_obs
                                        rmse = np.sqrt(np.mean(residuals**2))
                                        mean_obs = np.mean(binned_obs)
                                        nrmsd = (rmse / mean_obs) * 100 if mean_obs > 0 else 0
                                        dof = len(binned_obs)
                                        sigma = np.where(binned_err == 0, 1e-10, binned_err)
                                        reduced_chi_square = np.sum((residuals / sigma)**2) / dof if dof > 0 else float('inf')

                                        print(f"  [{st['label']}] RMSE: {rmse:.4e}, NRMSD: {nrmsd:.2f}%, Reduced χ²: {reduced_chi_square:.4f}")
                            else:
                                print(f"\n--- Binned Fit Evaluation: {csv_label} ---")
                                print("  エラー: 有効なTAA領域にデータがありません。")

                except Exception as e:
                    print(f"CSV error: {e}")

        # --- 軸の同期処理 ---
        if PLOT_MODE != "CSV_ONLY" and not CSV_USE_SHARED_Y_AXIS and has_csv_plot:
            def get_align_params(taa_arr, val_arr):
                if len(taa_arr) == 0: return 0, 1
                val_max = np.max(val_arr)
                diff = np.abs(np.mod(taa_arr, 360.0))
                diff = np.minimum(diff, 360.0 - diff)
                idx_peri = np.argmin(diff)
                return val_arr[idx_peri], val_max

            # 代表として1つ目のモデルを使用して比率を決定
            ref_data = all_models_data[0]["data"]
            ref_taa = all_models_data[0]["taa"]
            if isinstance(ref_data, dict):
                sim_m = np.max([np.max(v) for v in ref_data.values()])
                sim_p, _ = get_align_params(ref_taa, list(ref_data.values())[0])
            else:
                sim_p, sim_m = get_align_params(ref_taa, ref_data)

            obs_vals_list, obs_taas_list = [], []
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

            ratio = obs_p / sim_p if sim_p > 0 else 1.0
            print(f"\nAlign Ratio (Obs/Sim at Perihelion using {all_models_data[0]['setting']['label']}): {ratio:.4f}")

            # 複数モデル全体の最大値を考慮
            global_sim_m = max([np.max(list(m["data"].values())) if isinstance(m["data"], dict) else np.max(m["data"]) for m in all_models_data])
            
            required_sim_top = max(global_sim_m, obs_m / ratio)
            final_sim_top = required_sim_top * 1.1
            final_obs_top = final_sim_top * ratio

            ax1.set_ylim(0, final_sim_top)
            target_ax.set_ylim(0, final_obs_top)
            target_ax.grid(False)

        if PLOT_MODE == "CSV_ONLY" and y2_max_data > 0:
            ax1.set_ylim(0, y2_max_data * 1.1)

    # =======================================================
    # 生データを用いた厳密な総量比較 (時間加重平均)
    # =======================================================
    if PLOT_MODE != "CSV_ONLY" and len(all_models_data) > 0 and has_csv_plot:
        print("\n=== Unbinned Total Amount Comparison (Time-weighted) ===")
        MERCURY_ECCENTRICITY = 0.20563593

        for csv_setting in CSV_SETTINGS:
            csv_type = csv_setting.get("type", "UNKNOWN")
            if CSV_PLOT_SELECTION != "BOTH" and csv_type != CSV_PLOT_SELECTION:
                continue

            csv_path = csv_setting["path"]
            if not os.path.exists(csv_path): continue

            try:
                try:
                    df = pd.read_csv(csv_path, encoding='shift_jis')
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding='cp932')
                    
                csv_taa_raw = df.iloc[:, 2].values
                csv_density_raw = df.iloc[:, 3].values * 1e11
                sort_idx = np.argsort(csv_taa_raw)
                obs_taa = csv_taa_raw[sort_idx]
                obs_dens = csv_density_raw[sort_idx]

                theta_rad = np.radians(obs_taa)
                weights = 1.0 / (1.0 + MERCURY_ECCENTRICITY * np.cos(theta_rad))**2

                mean_obs_density = np.average(obs_dens, weights=weights)
                total_atoms_obs = mean_obs_density * NORMALIZATION_AREA_HALF_CM2

                print(f"\n[{csv_setting['label']}]")
                print(f"  Observed Time-Weighted Total : {total_atoms_obs:.4e} [atoms]")
                print("-" * 55)

                for mod_info in all_models_data:
                    st = mod_info["setting"]
                    sim_taa = mod_info["taa"]
                    sim_data = mod_info["data"]

                    if isinstance(sim_data, dict) and csv_type in sim_data:
                        mod_dens_full = sim_data[csv_type]
                    elif not isinstance(sim_data, dict):
                        mod_dens_full = sim_data
                    else:
                        continue 

                    mod_dens_matched = np.interp(obs_taa, sim_taa, mod_dens_full)
                    mean_mod_density = np.average(mod_dens_matched, weights=weights)
                    total_atoms_mod = mean_mod_density * NORMALIZATION_AREA_HALF_CM2

                    diff_atoms = total_atoms_mod - total_atoms_obs
                    diff_percent = (diff_atoms / total_atoms_obs) * 100 if total_atoms_obs > 0 else 0
                    ratio = total_atoms_mod / total_atoms_obs if total_atoms_obs > 0 else 0

                    print(f"  Model: {st['label']}")
                    print(f"    Total Amount    : {total_atoms_mod:.4e} [atoms]")
                    print(f"    Difference (M-O): {diff_atoms:+.4e} [atoms] ({diff_percent:+.2f}%)")
                    print(f"    Ratio (Mod/Obs) : {ratio:.3f}")
                print("=" * 55)

            except Exception as e:
                print(f"Total Amount Calculation Error for {csv_type}: {e}")

    # =======================================================
    # 最終レイアウト調整と表示
    # =======================================================
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

    if fig_binned is not None:
        ax_binned.set_ylim(bottom=0)
        if SHOW_LEGEND:
            ax_binned.legend(loc='upper left', fontsize=12)
        fig_binned.tight_layout()

    plt.show()

else:
    print("データ処理に失敗したか、有効なデータがなかったため終了します。")