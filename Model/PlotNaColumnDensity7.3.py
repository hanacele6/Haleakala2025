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

# 軌道離心率 (時間重み付けに使用)
MERCURY_ECCENTRICITY = 0.20563593

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

# ★★★ グリッド読み込み設定 ★★★
USE_COUNT_GRID = False  #True Count False Grid

# ★★★ グリッド設定 (デフォルト値)
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★★★ 自動スキャンと完全一致フィルタリングの設定 ★★★
BASE_RESULT_DIR = r"./SimulationResult_202607"

# 🌟 複数ヒットを防ぐため、フォルダ名（ファイル名）の「完全一致」で指定
TARGET_MODELS = {
    #"ParabolicHop_72x36_NoEq_DT100_0622_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1res_season2",
    #"ParabolicHop_72x36_NoEq_DT100_0623_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_3res_season2",
    #"ParabolicHop_72x36_NoEq_DT100_0624_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_5res_season2",
    #"ParabolicHop_72x36_NoEq_DT100_0622_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_Test_3",
    #"ParabolicHop_72x36_NoEq_DT100_0622_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_Test_old3",
    #"ParabolicHop_72x36_NoEq_DT100_0626_Multi_BD0.5_U1.85_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_Default":"Default",
    #"ParabolicHop_72x36_NoEq_DT100_0629_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_TrueG": "Bestfit" ,
    "ParabolicHop_72x36_NoEq_DT100_0703_BD0.5_UF1.85_Q2.0_A2.0e+07_LT190k_15yr":"StrongDiffusion",
}

# カラーパレットとマーカーの自動割り当てセット
COLOR_PALETTE = [
    {"dawn": "#1a365d", "dusk": "#742a2a"},  # 1つ目
    {"dawn": "#2b6cb0", "dusk": "#c53030"},  # 2つ目
    {"dawn": "#4299e1", "dusk": "#f56565"},  # 3つ目
    {"dawn": "#90cdf4", "dusk": "#feb2b2"},  # 4つ目
    {"dawn": "#4a5568", "dusk": "#a0aec0"},  # 5つ目
]
MARKERS = ["o", "s", "^", "D", "v", "x"]


# ★★★ フィッティング評価設定 ★★★
EVALUATE_FIT = False  
BIN_SIZE_DEG = 1.0  
SHOW_ERROR_BARS = True  
EXCLUDE_TAA_RANGES = []

# ★★★ プロット対象のシミュレーション年 ★★★
TARGET_YEAR = 3

# ★★★ 年間比較・統計量の確認設定 ★★★
# Trueにすると、指定した年の平均値と「普通の標準偏差」、および年間の差分を出力します
CALCULATE_YEAR_DIFF = True  
DIFF_TARGET_YEARS = [10, 11, 12, 13, 14, 15]  # 順番に隣り合う年を比較します (例: 3と4、4と5)

# ★★★ プロットモード選択 (シミュレーション側)
# 選択肢: "ALL", "DAYSIDE_TOTAL", "DAWN", "DUSK", "DAWN_OUTER", "DUSK_OUTER", "CSV_ONLY"
PLOT_MODE = "ALL"

# ★★★ CSVプロット選択モード (観測データ側)
# "DAWN" : DawnのCSVのみ表示, "DUSK" : DuskのCSVのみ表示, "BOTH" : 両方表示
CSV_PLOT_SELECTION = "BOTH"

# ★★★ 軸ラベル名 (共通)
COMMON_Y_LABEL = "Column Density [atoms/cm²]"

# ★★★ 凡例を表示するか
SHOW_LEGEND = True

# ★★★ 比較用CSVファイルの設定
SHOW_CSV_OVERLAY = True  
CSV_USE_SHARED_Y_AXIS = True  

# ★★★ 複数CSVの設定リスト
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


# --- 3. モデルセッティングの自動構築 (完全一致判定版) ---
MODEL_SETTINGS = []
if os.path.exists(BASE_RESULT_DIR) and PLOT_MODE != "CSV_ONLY":
    # 親フォルダ直下のサブディレクトリをスキャン
    subdirs = [d for d in os.listdir(BASE_RESULT_DIR) if os.path.isdir(os.path.join(BASE_RESULT_DIR, d))]
    subdirs.sort()
    
    model_idx = 0
    for folder in subdirs:
        # TARGET_MODELS のキーにフォルダ名が含まれているかチェック
        if TARGET_MODELS:
            if folder not in TARGET_MODELS:
                continue
        
        full_path = os.path.join(BASE_RESULT_DIR, folder)
        
        # 解像度サフィックスの判定
        current_res = GRID_RESOLUTION  # デフォルト値（101）
        if "1res" in folder:
            label_suffix = " (101³)"
            current_res = 101
        elif "3res" in folder:
            label_suffix = " (301³)"
            current_res = 301
        elif "5res" in folder:
            label_suffix = " (501³)"
            current_res = 501
        else:
            label_suffix = ""

        custom_label = TARGET_MODELS.get(folder, folder[:15])
        label = f"{custom_label}{label_suffix}"
        
        colors = COLOR_PALETTE[model_idx % len(COLOR_PALETTE)]
        marker = MARKERS[model_idx % len(MARKERS)]
        
        MODEL_SETTINGS.append({
            "dir": full_path,
            "label": label,  # ここに個別設定されたラベルが入る
            "color_dawn": colors["dawn"],
            "color_dusk": colors["dusk"],
            "marker_dawn": marker,
            "marker_dusk": marker,
            "grid_res": current_res,
            "max_rm": GRID_MAX_RM,
            "color_single": colors["dawn"],
            "marker_single": marker,
            "alpha": 1.0,
            "zorder": model_idx + 1
        })
        model_idx += 1

print(f"自動検出・構築された描画対象モデル数: {len(MODEL_SETTINGS)}")


# --- 4. データ処理関数 ---
def process_simulation_data(target_dir, mode, target_year, grid_res, max_rm, use_count_grid):
    grid_total_width_m = 2 * max_rm * RM_m
    cell_size_m = grid_total_width_m / grid_res
    cell_volume_m3 = cell_size_m ** 3
    
    mid_index_x = (grid_res - 1) // 2
    mid_index_y = (grid_res - 1) // 2
    quarter_index_offset = mid_index_y // 2
    idx_dawn_outer_limit = quarter_index_offset
    idx_dusk_outer_start = (grid_res - 1) - quarter_index_offset

    file_prefix = 'atom_count_grid_' if use_count_grid else 'density_grid_'

    try:
        all_files = [f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith(file_prefix)]
        if not all_files:
            print(f"エラー: ディレクトリ '{target_dir}' に有効なプレフィックス '{file_prefix}' の .npy ファイルがありません。")
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

    for filename, time_h, taa in tqdm(filtered_files, desc=f"Loading {os.path.basename(target_dir)[:15]} (Y{target_year})..."):
        filepath = os.path.join(target_dir, filename)
        loaded_grid = np.load(filepath)

        dayside_grid = loaded_grid[mid_index_x:, :, :]

        if use_count_grid:
            atoms_grid = dayside_grid
        else:
            atoms_grid = dayside_grid * cell_volume_m3
            atoms_grid[0, :, :] *= 0.5

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

if PLOT_MODE != "CSV_ONLY":
    print(f"処理モード: {PLOT_MODE} (データ元: {'Count Grid' if USE_COUNT_GRID else 'Density Grid'})")
    year_str = f"Year {TARGET_YEAR}" if TARGET_YEAR != "ALL" else "All Years"
    print(f"対象データ: {year_str}")
    
    for mod_set in MODEL_SETTINGS:
        g_res = mod_set["grid_res"]
        g_rm = mod_set.get("max_rm", 5.0)
        
        taa, data = process_simulation_data(mod_set["dir"], PLOT_MODE, TARGET_YEAR, g_res, g_rm, USE_COUNT_GRID)

        if taa is not None:
            all_models_data.append({
                "setting": mod_set,
                "taa": taa,
                "data": data
            })

if len(all_models_data) > 0 or PLOT_MODE == "CSV_ONLY":
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
    ax1.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')  
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))
    
    for mod_info in all_models_data:
        st = mod_info["setting"]
        sim_taa = mod_info["taa"]
        sim_data = mod_info["data"]
        mod_alpha = st.get("alpha", 0.8)
        mod_zorder = st.get("zorder", 3)
        
        if PLOT_MODE == "ALL":
            if "DAWN" in sim_data and len(sim_data["DAWN"]) > 0:
                ax1.plot(sim_taa, sim_data["DAWN"], color=st.get("color_dawn", "blue"), label=f"{st['label']}: Dawn", marker=st.get("marker_dawn", "^"), markersize=6, alpha=mod_alpha, zorder=mod_zorder, linestyle='None')
            if "DUSK" in sim_data and len(sim_data["DUSK"]) > 0:
                ax1.plot(sim_taa, sim_data["DUSK"], color=st.get("color_dusk", "red"), label=f"{st['label']}: Dusk", marker=st.get("marker_dusk", "v"), markersize=6, alpha=mod_alpha, zorder=mod_zorder, linestyle='None')
        else:
            p_color = st.get("color_dawn", "blue") if "DAWN" in PLOT_MODE else (st.get("color_dusk", "red") if "DUSK" in PLOT_MODE else st.get("color_single", "blue"))
            p_marker = st.get("marker_dawn", "^") if "DAWN" in PLOT_MODE else (st.get("marker_dusk", "v") if "DUSK" in PLOT_MODE else st.get("marker_single", "o"))
            ax1.plot(sim_taa, sim_data, label=f"{st['label']}: {PLOT_MODE}", color=p_color, alpha=mod_alpha, zorder=mod_zorder, marker=p_marker, linestyle='None')

    # --- CSVオーバーレイ処理 ---
    y2_max_data = 0
    has_csv_plot = False
    fig_binned, ax_binned = None, None

    if SHOW_CSV_OVERLAY or PLOT_MODE == "CSV_ONLY":
        target_ax = ax1 if (PLOT_MODE == "CSV_ONLY" or CSV_USE_SHARED_Y_AXIS) else ax1.twinx()
        if target_ax != ax1:
            target_ax.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')
            target_ax.tick_params(axis='y', labelcolor='black')

        for csv_setting in CSV_SETTINGS:
            csv_type = csv_setting.get("type", "UNKNOWN")
            if CSV_PLOT_SELECTION != "BOTH" and csv_type != CSV_PLOT_SELECTION: continue 
            csv_path = csv_setting["path"]
            csv_label = csv_setting["label"]
            csv_color = csv_setting.get("color", "green")
            csv_marker = csv_setting.get("marker", "x")

            if os.path.exists(csv_path):
                try:
                    try: df = pd.read_csv(csv_path, encoding='shift_jis')
                    except UnicodeDecodeError: df = pd.read_csv(csv_path, encoding='cp932')

                    if df.shape[1] >= 4:
                        csv_taa = df.iloc[:, 2].values
                        csv_density = df.iloc[:, 3].values * 1e11
                        csv_error = df.iloc[:, 4].values * 1e10 if df.shape[1] >= 5 else None

                        if SHOW_ERROR_BARS and csv_error is not None:
                            target_ax.errorbar(csv_taa, csv_density, yerr=csv_error, label=csv_label, color=csv_color, ecolor='black', fmt=csv_marker, markersize=6, capsize=2, elinewidth=1.0, alpha=1.0, zorder=2, linestyle='None')
                        else:
                            target_ax.scatter(csv_taa, csv_density, label=csv_label, color=csv_color, marker=csv_marker, s=40, zorder=2, alpha=1.0) 
                        if len(csv_density) > 0: y2_max_data = max(y2_max_data, np.max(csv_density))
                        has_csv_plot = True

                        # フィッティング評価 (EVALUATE_FIT) ループ
                        if EVALUATE_FIT and PLOT_MODE == "ALL" and len(all_models_data) > 0:
                            bin_edges = np.arange(0, 360 + BIN_SIZE_DEG, BIN_SIZE_DEG)
                            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                            binned_obs, binned_err, valid_bin_centers = [], [], []
                            
                            for i in range(len(bin_centers)):
                                b_start, b_end, b_center = bin_edges[i], bin_edges[i+1], bin_centers[i]
                                if any(t_start <= b_center <= t_end for t_start, t_end in EXCLUDE_TAA_RANGES): continue
                                mask = (csv_taa >= b_start) & (csv_taa < b_end)
                                if np.any(mask):
                                    obs_mean = np.mean(csv_density[mask])
                                    err_mean = np.sqrt(np.sum(csv_error[mask]**2)/len(csv_density[mask]) + np.var(csv_density[mask], ddof=1)) if csv_error is not None and len(csv_density[mask])>1 else (csv_error[mask][0] if csv_error is not None else obs_mean*0.1)
                                    binned_obs.append(obs_mean)
                                    binned_err.append(err_mean)
                                    valid_bin_centers.append(b_center)

                            binned_obs, binned_err, valid_bin_centers = np.array(binned_obs), np.array(binned_err), np.array(valid_bin_centers)
                            if len(binned_obs) > 0:
                                if fig_binned is None:
                                    fig_binned, ax_binned = plt.subplots(figsize=(10, 7))
                                    ax_binned.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
                                    ax_binned.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')
                                    ax_binned.set_xlim(0, 360)
                                    ax_binned.set_xticks(np.arange(0, 361, 60))
                                if SHOW_ERROR_BARS:
                                    ax_binned.errorbar(valid_bin_centers, binned_obs, yerr=binned_err, xerr=BIN_SIZE_DEG/2.0, label=f"Obs Binned: {csv_type}", color='black', ecolor='black', fmt='o', markersize=5, capsize=3, elinewidth=1.0, alpha=1.0, zorder=4, linestyle='None')
                                ax_binned.scatter(valid_bin_centers, binned_obs, color=csv_color, marker='o', s=50, edgecolors='black', linewidths=0.5, zorder=5)

                                print(f"\n--- Binned Fit Evaluation: {csv_label} ---")
                                for mod_info in all_models_data:
                                    st, sim_taa, sim_data = mod_info["setting"], mod_info["taa"], mod_info["data"]
                                    if csv_type in sim_data:
                                        binned_mod = [np.mean(np.interp(np.linspace(bc-BIN_SIZE_DEG/2.0, bc+BIN_SIZE_DEG/2.0, 20), sim_taa, sim_data[csv_type])) for bc in valid_bin_centers]
                                        binned_mod = np.array(binned_mod)
                                        
                                        step_x, step_y = [], []
                                        for j, bc in enumerate(valid_bin_centers):
                                            step_x.extend([bc - BIN_SIZE_DEG/2.0, bc + BIN_SIZE_DEG/2.0])
                                            step_y.extend([binned_mod[j], binned_mod[j]])
                                            if j < len(valid_bin_centers)-1 and not np.isclose(bc+BIN_SIZE_DEG, valid_bin_centers[j+1]):
                                                step_x.append(np.nan); step_y.append(np.nan)
                                        ax_binned.plot(step_x, step_y, label=f"{st['label']}: {csv_type}", color=st.get("color_dawn") if csv_type=="DAWN" else st.get("color_dusk"), zorder=3, linestyle='-', linewidth=1.5, alpha=st.get("alpha", 0.8))
                                        residual = binned_mod - binned_obs
                                        rmse = np.sqrt(np.mean(residual**2))
                                        print(f"  [{st['label']}] RMSE: {rmse:.4e}, NRMSD: {(rmse/np.mean(binned_obs))*100:.2f}%")
                except Exception as e: print(f"CSV error: {e}")

        if PLOT_MODE != "CSV_ONLY" and not CSV_USE_SHARED_Y_AXIS and has_csv_plot:
            ax1.set_ylim(bottom=0)
        if PLOT_MODE == "CSV_ONLY" and y2_max_data > 0: ax1.set_ylim(0, y2_max_data * 1.1)

    # =======================================================
    # モデル間定量比較 (1個目のモデルを基準ベースラインにする)
    # =======================================================
    if PLOT_MODE != "CSV_ONLY" and len(all_models_data) > 0:
        print("\n=== Unbinned Time-Weighted Average Comparison (Baseline: 1st Model) ===")
        
        base_mod_info = all_models_data[0]
        base_st = base_mod_info["setting"]
        base_taa = base_mod_info["taa"]
        base_data = base_mod_info["data"]
        
        target_types = ["DAWN", "DUSK"] if PLOT_MODE == "ALL" else [PLOT_MODE]
        
        for t_type in target_types:
            print(f"\n[Target Region: {t_type}]")
            
            base_dens_full = base_data[t_type] if isinstance(base_data, dict) else base_data
            weights = 1.0 / (1.0 + MERCURY_ECCENTRICITY * np.cos(np.radians(base_taa)))**2
            avg_dens_base = np.average(base_dens_full, weights=weights)
            
            if "OUTER" in t_type:
                current_area = NORMALIZATION_AREA_QUARTER_CM2
            elif t_type in ["DAWN", "DUSK"]:
                current_area = NORMALIZATION_AREA_HALF_CM2
            else:
                current_area = NORMALIZATION_AREA_CM2
                
            total_atoms_base = avg_dens_base * current_area
            
            print(f"  BASELINE Model: {base_st['label']}")
            print(f"    Baseline Avg Column Density : {avg_dens_base:.4e} [atoms/cm²]")
            print(f"    Baseline Total Atoms        : {total_atoms_base:.4e} [atoms]")
            print("-" * 75)
            
            for mod_info in all_models_data:
                st = mod_info["setting"]
                sim_taa = mod_info["taa"]
                sim_data = mod_info["data"]
                
                mod_dens_full = sim_data[t_type] if isinstance(sim_data, dict) else sim_data
                
                interp_mod_dens = np.interp(base_taa, sim_taa, mod_dens_full)
                avg_dens_mod = np.average(interp_mod_dens, weights=weights)
                total_atoms_mod = avg_dens_mod * current_area
                
                ratio = avg_dens_mod / avg_dens_base if avg_dens_base > 0 else 0.0
                
                print(f"  Model: {st['label']}")
                print(f"    Avg Column Density : {avg_dens_mod:.4e} [atoms/cm²] (Ratio vs Base: {ratio:.3f})")
                print(f"    Total Atoms        : {total_atoms_mod:.4e} [atoms] (Ratio vs Base: {ratio:.3f})")

    # =======================================================
    # 🌟 新機能：指定した年の「普通の標準偏差」および年間の差分を評価
    # =======================================================
    if CALCULATE_YEAR_DIFF and PLOT_MODE != "CSV_ONLY" and len(MODEL_SETTINGS) > 0:
        print("\n=== Yearly Statistical Analysis (Steady-State & Variation Check) ===")
        common_taa = np.linspace(0, 360, 361)
        common_weights = 1.0 / (1.0 + MERCURY_ECCENTRICITY * np.cos(np.radians(common_taa)))**2

        target_types = ["DAWN", "DUSK"] if PLOT_MODE == "ALL" else [PLOT_MODE]

        for mod_set in MODEL_SETTINGS:
            print(f"\n[Model: {mod_set['label']}]")
            g_res = mod_set["grid_res"]
            g_rm = mod_set.get("max_rm", 5.0)

            yearly_data = {}
            for yr in DIFF_TARGET_YEARS:
                y_taa, y_data = process_simulation_data(mod_set["dir"], PLOT_MODE, yr, g_res, g_rm, USE_COUNT_GRID)
                if y_taa is not None and len(y_taa) > 0:
                    yearly_data[yr] = {"taa": y_taa, "data": y_data}
            
            for i in range(len(DIFF_TARGET_YEARS) - 1):
                y1 = DIFF_TARGET_YEARS[i]
                y2 = DIFF_TARGET_YEARS[i+1]

                if y1 not in yearly_data or y2 not in yearly_data:
                    print(f"  --- Comparison: Year {y1} vs Year {y2} はデータ不足のためスキップ ---")
                    continue

                print(f"  --- Comparison: Year {y1} vs Year {y2} ---")
                
                for t_type in target_types:
                    d1_raw = yearly_data[y1]["data"][t_type] if isinstance(yearly_data[y1]["data"], dict) else yearly_data[y1]["data"]
                    d2_raw = yearly_data[y2]["data"][t_type] if isinstance(yearly_data[y2]["data"], dict) else yearly_data[y2]["data"]
                    
                    d1_interp = np.interp(common_taa, yearly_data[y1]["taa"], d1_raw)
                    d2_interp = np.interp(common_taa, yearly_data[y2]["taa"], d2_raw)

                    # 1. 時間重み付き平均
                    avg_y1 = np.average(d1_interp, weights=common_weights)
                    avg_y2 = np.average(d2_interp, weights=common_weights)
                    
                    # 2. 普通の標準偏差（軌道上の密度のバラつき = 季節変動の大きさ）
                    std_y1 = np.std(d1_interp)
                    std_y2 = np.std(d2_interp)

                    # 3. 年間の差分 (収束確認用)
                    avg_diff_pct = (avg_y2 - avg_y1) / avg_y1 * 100.0 if avg_y1 > 0 else 0.0
                    mean_abs_diff = np.mean(np.abs(d2_interp - d1_interp))

                    print(f"    Region: {t_type}")
                    print(f"      Year {y1} - Time-Weighted Avg : {avg_y1:.4e} [atoms/cm²]")
                    print(f"                Standard Deviation  : {std_y1:.4e} (軌道1周でのバラつき)")
                    print(f"      Year {y2} - Time-Weighted Avg : {avg_y2:.4e} [atoms/cm²]")
                    print(f"                Standard Deviation  : {std_y2:.4e} (軌道1周でのバラつき)")
                    print(f"      => TWA Difference (Y{y2} - Y{y1}): {avg_y2 - avg_y1:.4e} ({avg_diff_pct:+.3f} %)")
                    print(f"      => Mean Absolute Diff (Y{y2} - Y{y1}): {mean_abs_diff:.4e} [atoms/cm²]")


    if SHOW_LEGEND:
        if PLOT_MODE != "CSV_ONLY" and 'target_ax' in locals() and target_ax != ax1:
            l1, lb1 = ax1.get_legend_handles_labels()
            l2, lb2 = target_ax.get_legend_handles_labels()
            ax1.legend(l1 + l2, lb1 + lb2, loc='upper left', fontsize=12)
        else: ax1.legend(loc='upper left', fontsize=12)
    fig.tight_layout()
    if fig_binned is not None: ax_binned.legend(loc='upper left'); fig_binned.tight_layout()
    plt.show()