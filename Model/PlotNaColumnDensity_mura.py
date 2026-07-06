# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, RadioButtons
from tqdm import tqdm
import re
from scipy.interpolate import CubicSpline
import pandas as pd

# --- 1. 物理定数と正規化因子 ---
RM_m = 2.440e6  # 水星の半径 [m]
CM_PER_M = 100.0
RM_cm = RM_m * CM_PER_M

# ====================================================================
# ★★★ 動的切り替えのための面積定義 (単位: cm^2) ★★★
# ====================================================================
AREA_SCHEMES = {
    "1.3 Rm Disk (Mura)": np.pi * ((1.3 * RM_cm) ** 2),  # 約 3.167e17 cm^2
    "Half Surface": 2 * np.pi * (RM_cm ** 2),            # 水星表面積の半分 (約 3.748e17 cm^2)
    "Quarter Surface": np.pi * (RM_cm ** 2),              # 水星表面積の半分の半分 (約 1.874e17 cm^2)
    "10x10 arcsec (Potter)": (100.0 / 9.0) * (RM_cm ** 2)
}

# データ処理時のベースとなる面積（Mura 2023準拠）
BASE_TOTAL_AREA_CM2 = AREA_SCHEMES["1.3 Rm Disk (Mura)"]
BASE_HALF_AREA_CM2 = BASE_TOTAL_AREA_CM2 / 2.0
BASE_QUARTER_AREA_CM2 = BASE_TOTAL_AREA_CM2 / 4.0

MERCURY_YEAR_HOURS = 87.969 * 24 

# --- 2. ユーザー設定 ---
SHOW_MURA_2023 = False
GRID_RESOLUTION = 101
GRID_MAX_RM = 5.0

# ★★★ シミュレーション結果のディレクトリ
output_dir = r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0609_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1e24"

# ★★★ フィッティング評価設定
EVALUATE_FIT = False
BIN_SIZE_DEG = 30.0
SHOW_ERROR_BARS = True
EXCLUDE_TAA_RANGES = []

TARGET_YEAR = 3
PLOT_MODE = "DUSK"
CSV_PLOT_SELECTION = "DUSK"
COMMON_Y_LABEL = "Column Density [atoms/cm²]"
SHOW_LEGEND = True
SHOW_CSV_OVERLAY = True
CSV_USE_SHARED_Y_AXIS = True

CSV_SETTINGS = [
    {
        "path": r"C:\Users\hanac\univ\Mercury/DAWN.csv",
        "label": "Observation: Dawn", "color": "green", "marker": "x", "type": "DAWN"
    },
    {
        "path": r"C:\Users\hanac\univ\Mercury/DUSK.csv",
        "label": "Observation: Dusk", "color": "magenta", "marker": "+", "type": "DUSK"
    },
    {
        "path": r"C:\Users\hanac\univ\Mercury/Potter2007_DUSK.csv", 
        "label": "Potter(DUSK)", "color": "orange", "marker": "o", "type": "ref"
    },
    #{
    #    "path": r"C:\Users\hanac\univ\Mercury/Potter2007_DAWN.csv", 
    #    "label": "Potter(DAWN)", "color": "cyan", "marker": "o", "type": "ref"
    #}
]

# --- 2.5 Mura 2023 モデルデータ生成関数 ---
def get_mura_2023_curve(taa_array):
    taa_points = np.array([0, 30, 60, 75, 90, 120, 150, 180, 210, 240, 270, 290, 300, 330, 360])
    cd_points = np.array([
        1.26, 0.95, 0.78, 0.75, 0.80, 0.98, 1.30, 1.68, 
        1.42, 1.10, 0.90, 0.82, 0.83, 1.00, 1.26
    ]) * 1e11
    cs = CubicSpline(taa_points, cd_points, bc_type='periodic')
    return cs(taa_array)

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
def process_simulation_data(target_dir, mode, target_year):
    try:
        all_files = [f for f in os.listdir(target_dir) if f.endswith('.npy') and f.startswith('density_grid_')]
        if not all_files: return None, None
    except FileNotFoundError: return None, None

    filtered_files = []
    for f in all_files:
        match = re.search(r'_t(\d+)_taa(\d+)\.npy$', f)
        if match:
            time_h, taa = int(match.group(1)), int(match.group(2))
            file_year = int(time_h // MERCURY_YEAR_HOURS) + 1
            if target_year != "ALL" and file_year != target_year: continue
            filtered_files.append((f, time_h, taa))
            
    if not filtered_files: return None, None
    filtered_files.sort(key=lambda x: x[1])

    sim_results_taa = []
    results_dict = {"DAWN": [], "DUSK": []}
    single_result_density = []

    for filename, time_h, taa in tqdm(filtered_files, desc=f"Processing Year {target_year}"):
        filepath = os.path.join(target_dir, filename)
        density_grid_m3 = np.load(filepath)
        dayside_grid = density_grid_m3[mid_index_x:, :, :]
        atoms_grid = dayside_grid * cell_volume_m3
        atoms_grid[0, :, :] *= 0.5

        sum_mid = 0
        if mode in ["DAWN", "DUSK", "ALL"]:
            sum_mid = np.sum(atoms_grid[:, mid_index_y, :])

        if mode == "ALL":
            sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
            dens_dawn = (sum_dawn + 0.5 * sum_mid) / BASE_HALF_AREA_CM2
            results_dict["DAWN"].append(dens_dawn)

            sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
            dens_dusk = (sum_dusk + 0.5 * sum_mid) / BASE_HALF_AREA_CM2
            results_dict["DUSK"].append(dens_dusk)
        else:
            total_atoms, target_area = 0.0, 1.0
            if mode == "DAYSIDE_TOTAL":
                total_atoms = np.sum(atoms_grid)
                target_area = BASE_TOTAL_AREA_CM2
            elif mode == "DAWN":
                sum_dawn = np.sum(atoms_grid[:, :mid_index_y, :])
                total_atoms, target_area = sum_dawn + 0.5 * sum_mid, BASE_HALF_AREA_CM2
            elif mode == "DUSK":
                sum_dusk = np.sum(atoms_grid[:, mid_index_y + 1:, :])
                total_atoms, target_area = sum_dusk + 0.5 * sum_mid, BASE_HALF_AREA_CM2

            single_result_density.append(total_atoms / target_area)

        sim_results_taa.append(taa)

    sim_results_taa = np.array(sim_results_taa)
    sorted_idx = np.argsort(sim_results_taa)
    
    if mode == "ALL":
        return sim_results_taa[sorted_idx], {k: np.array(v)[sorted_idx] for k, v in results_dict.items()}
    else:
        return sim_results_taa[sorted_idx], np.array(single_result_density)[sorted_idx]

# --- 5. メイン処理とプロット ---
sim_results = {}
all_years = []

if PLOT_MODE != "CSV_ONLY":
    try:
        for f in os.listdir(output_dir):
            match = re.search(r'_t(\d+)_taa\d+\.npy$', f)
            if match:
                file_year = int(int(match.group(1)) // MERCURY_YEAR_HOURS) + 1
                if file_year not in all_years: all_years.append(file_year)
        all_years.sort()
    except FileNotFoundError: pass

    for y in (all_years if TARGET_YEAR == "ALL" else [TARGET_YEAR]):
        taa, data = process_simulation_data(output_dir, PLOT_MODE, y)
        if taa is not None and len(taa) > 0: sim_results[y] = {"taa": taa, "data": data}

    if len(sim_results) > 0:
        sorted_years = sorted(list(sim_results.keys()))
        sim_taa = np.concatenate([sim_results[y]["taa"] for y in sorted_years])
        sort_idx = np.argsort(sim_taa)
        sim_taa = sim_taa[sort_idx]
        
        if PLOT_MODE == "ALL":
            sim_data = {k: np.concatenate([sim_results[y]["data"][k] for y in sorted_years])[sort_idx] for k in ["DAWN", "DUSK"]}
        else:
            sim_data = np.concatenate([sim_results[y]["data"] for y in sorted_years])[sort_idx]

if sim_taa is not None or PLOT_MODE == "CSV_ONLY":
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_axes([0.24, 0.1, 0.73, 0.8])

    ax1.set_xlabel('True Anomaly Angle (deg)', fontsize=18)
    ax1.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.yaxis.get_offset_text().set_fontsize(14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(0, 360)
    ax1.set_xticks(np.arange(0, 361, 60))
    ax1.set_ylim(0, 2.8e11)
    
    artists_sim, artists_mura, artists_obs, artists_ref = [], [], [], []

    # --- シミュレーションデータのプロット ---
    if PLOT_MODE != "CSV_ONLY" and len(sim_results) > 0:
        sorted_years = sorted(list(sim_results.keys()))
        for idx, y in enumerate(sorted_years):
            s_taa, s_data = sim_results[y]["taa"], sim_results[y]["data"]
            intensity = 0.4 + 0.6 * (idx / max(1, len(sorted_years) - 1))
            
            if PLOT_MODE == "ALL":
                styles = {"DAWN": {"color": plt.get_cmap('Blues')(intensity) if len(sorted_years)>1 else 'blue', "marker": "^"},
                          "DUSK": {"color": plt.get_cmap('Reds')(intensity) if len(sorted_years)>1 else 'red', "marker": "v"}}
                for key, val_array in s_data.items():
                    line, = ax1.plot(s_taa, val_array, color=styles[key]["color"], label=f"Sim: {key} (Yr {y})",
                                     marker=styles[key]["marker"], markersize=6, alpha=0.8, linestyle='None') 
                    artists_sim.append(line)
            else:
                line, = ax1.plot(s_taa, s_data, label=f'Sim: {PLOT_MODE} (Yr {y})', 
                                 color=plt.get_cmap('Greens')(intensity) if len(sorted_years)>1 else 'blue', 
                                 alpha=0.8, marker='o', linestyle='None')
                artists_sim.append(line)

    for line in artists_sim:
        line.base_ydata = line.get_ydata().copy()

    target_ax = ax1 if (CSV_USE_SHARED_Y_AXIS or PLOT_MODE == "CSV_ONLY") else ax1.twinx()
    if target_ax != ax1:
        target_ax.set_ylabel(COMMON_Y_LABEL, fontsize=18, color='black')

    # --- Mura 2023 モデル ---
    if SHOW_MURA_2023:
        taa_dense = np.linspace(0, 360, 360)
        line, = target_ax.plot(taa_dense, get_mura_2023_curve(taa_dense), color='black', 
                               linestyle='-', linewidth=2.5, label='Mura (2023) Model', zorder=10)
        artists_mura.append(line)

    # --- CSVデータ読み込み ---
    has_csv_plot = False
    for csv_setting in CSV_SETTINGS:
        csv_type, csv_path, csv_label, csv_color, csv_marker = csv_setting.get("type", "UNKNOWN"), csv_setting["path"], csv_setting["label"], csv_setting.get("color", "green"), csv_setting.get("marker", "x")
        if csv_type in ["DAWN", "DUSK"] and CSV_PLOT_SELECTION != "BOTH" and csv_type != CSV_PLOT_SELECTION: continue 

        if os.path.exists(csv_path):
            try:
                try: 
                    df = pd.read_csv(csv_path, encoding='utf-8')
                except: 
                    try: 
                        df = pd.read_csv(csv_path, encoding='shift_jis')
                    except: 
                        df = pd.read_csv(csv_path, encoding='cp932')

                if df.shape[1] >= 4:
                    csv_taa, csv_density, csv_error = df.iloc[:, 2].values, df.iloc[:, 3].values * 1e11, df.iloc[:, 4].values * 1e10 if df.shape[1] >= 5 else None 
                    lbl = csv_label 
                else:
                    df = df.dropna()
                    csv_taa, csv_density, csv_error = df.iloc[:, 0].values, df.iloc[:, 1].values, None
                    if np.nanmax(csv_density) < 1e5: csv_density *= 1e11
                    lbl = csv_label

                tgt_list = artists_ref if csv_type == "ref" else artists_obs
                if SHOW_ERROR_BARS and csv_error is not None:
                    tgt_list.append(target_ax.errorbar(csv_taa, csv_density, yerr=csv_error, label=lbl, color=csv_color, ecolor='black', fmt=csv_marker, capsize=2, zorder=2, linestyle='None'))
                else:
                    tgt_list.append(target_ax.scatter(csv_taa, csv_density, label=lbl, color=csv_color, marker=csv_marker, s=40, zorder=2))
                has_csv_plot = True
            except Exception as e: print(f"CSV error: {e}")

    ax1.set_ylim(bottom=0)
    if target_ax != ax1: target_ax.set_ylim(bottom=0)

    # =======================================================
    # 凡例の動的更新関数 (オンのものだけを表示する)
    # =======================================================
    def update_legend():
        if not SHOW_LEGEND: return
        
        # 描画されているすべてのハンドル(マーカー/線)とラベルを取得
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = target_ax.get_legend_handles_labels() if target_ax != ax1 else ([], [])
        
        visible_handles = []
        visible_labels = []
        
        # 現在 visible=True になっているものだけを抽出
        for h, l in zip(handles1 + handles2, labels1 + labels2):
            is_visible = False
            if hasattr(h, 'lines'): # Errorbarなどのコンテナの場合
                is_visible = h.lines[0].get_visible() if len(h.lines) > 0 else False
            elif hasattr(h, 'get_visible'): # 通常のLineやScatterの場合
                is_visible = h.get_visible()
                
            if is_visible:
                visible_handles.append(h)
                visible_labels.append(l)

        # 現在の凡例を一度消去
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()
            
        # 表示対象がある場合のみ新しく凡例を描画
        if visible_handles:
            loc = 'upper left' if (PLOT_MODE != "CSV_ONLY" and not CSV_USE_SHARED_Y_AXIS and has_csv_plot) else 'upper right'
            ax1.legend(visible_handles, visible_labels, loc=loc, fontsize=12)

    # 初期状態の凡例を描画
    update_legend()

    # =======================================================
    # UI パネル 1: 表示のオン・オフトグル
    # =======================================================
    rax_toggle = fig.add_axes([0.02, 0.55, 0.14, 0.25]) 
    rax_toggle.set_facecolor('lightgoldenrodyellow')
    rax_toggle.set_title('Toggle Data', fontsize=11, fontweight='bold')
    check = CheckButtons(rax_toggle, ['sim', 'mura', 'obs', 'ref'], [True]*4)

    def toggle_visibility(label):
        targets = {'sim': artists_sim, 'mura': artists_mura, 'obs': artists_obs, 'ref': artists_ref}[label]
        
        # リストが空（データがない）場合は何もせずに終了してエラーを防ぐ
        if not targets: 
            return 
            
        # 現在の状態を取得して反転させる
        visible = not (targets[0].get_visible() if not hasattr(targets[0], 'lines') else targets[0].lines[0].get_visible())

        for artist in targets:
            if hasattr(artist, 'lines'):
                for el in [e for e in artist.lines if e is not None]:
                    if isinstance(el, (tuple, list)): [e.set_visible(visible) for e in el]
                    else: el.set_visible(visible)
            else: artist.set_visible(visible)
            
        # ★ ここで凡例を再描画する関数を呼び出す
        update_legend()
        fig.canvas.draw_idle()
        
    check.on_clicked(toggle_visibility)

    # =======================================================
    # UI パネル 2: 面積切り替えラジオボタン
    # =======================================================
    rax_radio = fig.add_axes([0.02, 0.25, 0.14, 0.25]) # 3項目になったので少し高さを調整
    rax_radio.set_facecolor('lightcyan')
    rax_radio.set_title('Norm. Area', fontsize=11, fontweight='bold')
    
    radio = RadioButtons(rax_radio, list(AREA_SCHEMES.keys()))

    def update_area(label):
        new_area = AREA_SCHEMES[label]
        scale_factor = BASE_TOTAL_AREA_CM2 / new_area 
        
        for line in artists_sim:
            line.set_ydata(line.base_ydata * scale_factor)
            
        ax1.relim()
        #ax1.autoscale_view(scalex=False, scaley=True)
        ax1.set_ylim(bottom=0)
        fig.canvas.draw_idle()

    radio.on_clicked(update_area)

    plt.show()

else:
    print("データ処理に失敗したため終了します。")