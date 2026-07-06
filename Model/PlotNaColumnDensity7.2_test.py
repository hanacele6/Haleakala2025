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
NORMALIZATION_AREA_HALF_CM2 = NORMALIZATION_AREA_CM2 /2.0
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
MODEL_SETTINGS = [
    {
         "dir": r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0602_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)",

        "label": "OldModel",
        "color_dawn": "cyan",
        "color_dusk": "orange",
        "marker_dawn": "^",
        "marker_dusk": "v",
        "grid_res": 101,  
        "max_rm": 5.0,
        "color_single": "blue",
        "marker_single": "o",
        "alpha": 1.0,
        "zorder": 5
    },
    #{
    #    "dir": r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0609_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1e24",
    #    "label": "NewModel",
    #    "color_dawn": "purple",    
    #    "color_dusk": "brown",     
    #    "marker_dawn": "^",
    #    "marker_dusk": "v",
    #    "grid_res": 101,  
    #    "max_rm": 5.0,
    #    "color_single": "red",  
    #    "marker_single": "x",
    #    "alpha": 0.6,
    #    "zorder": 6
    #},
]

# ★★★ 残差プロット(モデル間比較)設定 ★★★
SHOW_MODEL_RESIDUALS = True  # True: グラフ下段にモデル間の差分を表示, False: 表示しない
RESIDUAL_TYPE = "PERCENT"    # "PERCENT": ％で差分表示, "DIFF": 絶対値(atoms/cm2)で差分表示

# ★★★ フィッティング評価設定 ★★★
EVALUATE_FIT = False  # 最小二乗誤差(RMSE等)を計算するかどうか
BIN_SIZE_DEG = 1.0  # TAAのビン幅 [度] (例: 10度ごとに平均化)
SHOW_ERROR_BARS = True  # グラフに誤差棒を表示するか (True/False)
EXCLUDE_TAA_RANGES = []

# ★★★ プロット対象のシミュレーション年 (スピンアップ対応) ★★★
TARGET_YEAR = 1

# ★★★ プロットモード選択
PLOT_MODE = "DAWN"
CSV_PLOT_SELECTION = "DAWN"
COMMON_Y_LABEL = "Column Density [atoms/cm²]"
SHOW_LEGEND = True

# ★★★ 比較用CSVファイルの設定
SHOW_CSV_OVERLAY = True  
CSV_USE_SHARED_Y_AXIS = True  

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

mid_index_x = (GRID_RESOLUTION - 1) // 2
mid_index_y = (GRID_RESOLUTION - 1) // 2
quarter_index_offset = mid_index_y // 2
idx_dawn_outer_limit = quarter_index_offset
idx_dusk_outer_start = (GRID_RESOLUTION - 1) - quarter_index_offset


# --- 4. データ処理関数 ---
def process_simulation_data(target_dir, mode, target_year, grid_res, max_rm):
    grid_total_width_m = 2 * max_rm * RM_m
    cell_size_m = grid_total_width_m / grid_res
    cell_volume_m3 = cell_size_m ** 3
    
    mid_index_x = (grid_res - 1) // 2
    mid_index_y = (grid_res - 1) // 2
    quarter_index_offset = mid_index_y // 2
    idx_dawn_outer_limit = quarter_index_offset
    idx_dusk_outer_start = (grid_res - 1) - quarter_index_offset

    try:
        all_files = [f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')]
        if not all_files:
            return None, None
    except FileNotFoundError:
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
            
    if not filtered_files: return None, None
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
            dens_dawn = (sum_dawn + 0.5 * sum_mid) / NORMALIZATION_AREA_HALF_CM2
            results_dict["DAWN"].append(dens_dawn)

            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            dens_dusk = (sum_dusk + 0.5 * sum_mid) / NORMALIZATION_AREA_HALF_CM2
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
            
            single_result_density.append(total_atoms / target_area)
        sim_results_taa.append(taa)

    sim_results_taa = np.array(sim_results_taa)
    sorted_idx = np.argsort(sim_results_taa)
    sim_results_taa = sim_results_taa[sorted_idx]

    if mode == "ALL":
        for key in results_dict:
            results_dict[key] = np.array(results_dict[key])[sorted_idx]
        return sim_results_taa, results_dict
    else:
        return sim_results_taa, np.array(single_result_density)[sorted_idx]


# --- 5. メイン処理とプロット ---
all_models_data = []

if PLOT_MODE != "CSV_ONLY":
    for mod_set in MODEL_SETTINGS:
        taa, data = process_simulation_data(mod_set["dir"], PLOT_MODE, TARGET_YEAR, mod_set.get("grid_res", 101), mod_set.get("max_rm", 5.0))
        if taa is not None:
            all_models_data.append({"setting": mod_set, "taa": taa, "data": data})

if len(all_models_data) > 0 or PLOT_MODE == "CSV_ONLY":
    
    # 🌟 残差プロットモードのセットアップ
    if SHOW_MODEL_RESIDUALS and len(all_models_data) > 1 and PLOT_MODE != "CSV_ONLY":
        fig, (ax1, ax_res) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax_res.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
        if RESIDUAL_TYPE == "PERCENT":
            ax_res.set_ylabel('Difference [%]', fontsize=14, color='black')
        else:
            ax_res.set_ylabel('Diff [atoms/cm²]', fontsize=14, color='black')
        ax_res.tick_params(axis='both', which='major', labelsize=12)
        ax_res.grid(True, linestyle='--', alpha=0.6)
        ax_res.axhline(0, color='black', linewidth=1.5, linestyle='-')
    else:
        fig, ax1 = plt.subplots(figsize=(10, 7))
        ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
        ax_res = None

    ax1.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')  
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))

    y1_max_data = 0
    if PLOT_MODE != "CSV_ONLY":
        for i, mod_info in enumerate(all_models_data):
            st = mod_info["setting"]
            sim_taa = mod_info["taa"]
            sim_data = mod_info["data"]
            mod_alpha = st.get("alpha", 0.8)
            mod_zorder = st.get("zorder", 3)
            
            # --- メイングラフへのプロット ---
            if PLOT_MODE == "ALL":
                if "DAWN" in sim_data:
                    ax1.plot(sim_taa, sim_data["DAWN"], color=st.get("color_dawn", "blue"), label=f"{st['label']}: Dawn",
                             marker=st.get("marker_dawn", "^"), markersize=6, alpha=mod_alpha, zorder=mod_zorder, linestyle='None')
                if "DUSK" in sim_data:
                    ax1.plot(sim_taa, sim_data["DUSK"], color=st.get("color_dusk", "red"), label=f"{st['label']}: Dusk",
                             marker=st.get("marker_dusk", "v"), markersize=6, alpha=mod_alpha, zorder=mod_zorder, linestyle='None')
            else:
                p_color = st.get("color_single", "blue")
                p_marker = st.get("marker_single", "o")
                ax1.plot(sim_taa, sim_data, label=f"{st['label']}", color=p_color, alpha=mod_alpha, zorder=mod_zorder,
                         marker=p_marker, markersize=5, linestyle='None')
                y1_max_data = max(y1_max_data, np.max(sim_data))
                
            # 🌟 残差プロットの計算と描画 🌟
            # (1つ目のモデルを基準として、2つ目以降のモデルとの差分をプロットする)
            if ax_res is not None and i > 0:
                ref_taa = all_models_data[0]["taa"]
                ref_data = all_models_data[0]["data"]
                
                if PLOT_MODE == "ALL":
                    for region in ["DAWN", "DUSK"]:
                        if region in sim_data and region in ref_data:
                            # TAAが微妙にズレている場合を考慮して線形補間
                            ref_interp = np.interp(sim_taa, ref_taa, ref_data[region])
                            diff = sim_data[region] - ref_interp
                            
                            if RESIDUAL_TYPE == "PERCENT":
                                res_val = np.divide(diff, ref_interp, out=np.zeros_like(diff), where=ref_interp!=0) * 100
                            else:
                                res_val = diff
                                
                            c = st.get(f"color_{region.lower()}", "blue")
                            m = st.get(f"marker_{region.lower()}", "o")
                            ax_res.plot(sim_taa, res_val, color=c, marker=m, markersize=3, alpha=0.6, linestyle='None')
                else:
                    ref_interp = np.interp(sim_taa, ref_taa, ref_data)
                    diff = sim_data - ref_interp
                    
                    if RESIDUAL_TYPE == "PERCENT":
                        res_val = np.divide(diff, ref_interp, out=np.zeros_like(diff), where=ref_interp!=0) * 100
                    else:
                        res_val = diff
                        
                    c = st.get("color_single", "red")
                    m = st.get("marker_single", "x")
                    ax_res.plot(sim_taa, res_val, color=c, marker=m, markersize=4, alpha=0.7, linestyle='None', label=f"Residual: {st['label']}")

    # --- CSV観測データプロット (省略) ---
    y2_max_data = 0
    has_csv_plot = False

    if SHOW_CSV_OVERLAY or PLOT_MODE == "CSV_ONLY":
        if PLOT_MODE == "CSV_ONLY" or CSV_USE_SHARED_Y_AXIS:
            target_ax = ax1
        else:
            target_ax = ax1.twinx()
            target_ax.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')

        for csv_setting in CSV_SETTINGS:
            if CSV_PLOT_SELECTION != "BOTH" and csv_setting.get("type", "UNKNOWN") != CSV_PLOT_SELECTION:
                continue 

            csv_path = csv_setting["path"]
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding='shift_jis')
                csv_taa = df.iloc[:, 2].values
                csv_density = df.iloc[:, 3].values * 1e11
                if df.shape[1] >= 5:
                    csv_error = df.iloc[:, 4].values * 1e10
                else:
                    csv_error = None 

                if SHOW_ERROR_BARS and csv_error is not None:
                    target_ax.errorbar(csv_taa, csv_density, yerr=csv_error, label=csv_setting["label"], 
                                       color=csv_setting["color"], ecolor='black', fmt=csv_setting["marker"],
                                       markersize=6, capsize=2, elinewidth=1.0, alpha=1.0, zorder=2, linestyle='None')
                else:
                    target_ax.scatter(csv_taa, csv_density, label=csv_setting["label"], color=csv_setting["color"], 
                                      marker=csv_setting["marker"], s=40, zorder=2, alpha=1.0) 

    # --- 最終調整 ---
    ax1.set_ylim(bottom=0)
    if 'target_ax' in locals() and target_ax != ax1:
        target_ax.set_ylim(bottom=0)

    if SHOW_LEGEND:
        ax1.legend(loc='upper left', fontsize=12)
        if ax_res is not None:
            ax_res.legend(loc='upper left', fontsize=10)
            
    fig.tight_layout()
    plt.show()

else:
    print("データ処理に失敗したか、有効なデータがなかったため終了します。")