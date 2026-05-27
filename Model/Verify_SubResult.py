# -*- coding: utf-8 -*-
"""
シミュレーション結果 解析・可視化スクリプト (3年分フルタイム対応版)
- 100%積み上げの生成割合
- 生成と喪失(Lossの内訳)の絶対量
- 拡散によるビンごとのインベントリ推移
- 水星全体でのNa枯渇量（累積収支）の推移
- 【追加】ビンインベントリ推移の積み上げヒストグラム風表示
- 【追加】ビンインベントリ推移の2次元ヒートマップ表示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
import os
import glob
import re

# ==========================================
# 1. 設定
# ==========================================
RESULT_DIR = r"./SimulationResult_202605/ParabolicHop_72x36_NoEq_DT100_0517_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_V18"

#RESULT_DIR = r"./SimulationResult_202604/ParabolicHop_72x36_NoEq_DT100_0427_Multi_0.4Denabled_U1.85_Q0.27_Bouncetau30s_A0.5_LongLT"
#RESULT_DIR = r"./SimulationResult_202604/ParabolicHop_72x36_NoEq_DT100_0425_Multi_0.4Denabled_U1.85_Q0.27_Bouncetau30s_A1.0_LongLT"

# 解析対象のデータソース
# 'TimeSeries': 1年目からの全推移（スピンアップの立ち上がりや質量の増減を確認する場合）
# 'FinalYearAveraged': 最後の1年間のTAA平均（定常状態のバジェットを確認する場合）
#DATA_SOURCE = 'TimeSeries' 
DATA_SOURCE = 'FinalYearAveraged'

# 横軸の表示モード ('Time' または 'TAA')
# ※ TimeSeries かつ TAA を選んだ場合、3周分(0〜1080度...)の累積TAAとして自動的にアンラップされます。
X_AXIS_MODE = 'TAA' 

# ==========================================
# ヘルパー関数: モードに応じたデータの読み込みとTAAアンラップ
# ==========================================
def get_budget_data(target_dir, data_source, x_axis_mode):
    if data_source == 'FinalYearAveraged':
        csv_path = os.path.join(target_dir, "budget_statistics_per_taa.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSVが見つかりません: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.sort_values('TAA_Bin')
        
        # 最終年の集計データでTAAモードでない場合は警告を出す
        if x_axis_mode != 'TAA':
            print("警告: FinalYearAveraged は TAA のみの集計です。強制的に TAA で表示します。")
        return df, 'TAA_Bin', 'True Anomaly Angle (TAA) [deg]'
        
    else: # TimeSeries (3年分のログ)
        csv_path = os.path.join(target_dir, "budget_timeseries.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"時系列CSVが見つかりません: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if x_axis_mode == 'TAA':
            # 3年分のTAA(0~360のループ)を累積TAAに変換 (アンラップ処理)
            diff = df['TAA'].diff()
            wrap_count = (diff < -180).cumsum().fillna(0)
            df['Unwrapped_TAA'] = df['TAA'] + wrap_count * 360.0
            return df, 'Unwrapped_TAA', 'Cumulative TAA [deg]'
        else:
            return df, 'Time_hours', 'Simulation Time [hours]'

# ==========================================
# グラフ描画関数
# ==========================================
def plot_generation_ratio(target_dir, data_source, x_axis_mode):
    df, x_col, x_label = get_budget_data(target_dir, data_source, x_axis_mode)
    plt.figure(figsize=(10, 6))
    
    total_gen = df['Gen_PSD'] + df['Gen_TD'] + df['Gen_SWS'] + df['Gen_MMV']
    total_gen = total_gen.replace(0, 1e-10) # 0割りを防ぐ
    
    pct_td = df['Gen_TD'] / total_gen * 100
    pct_psd = df['Gen_PSD'] / total_gen * 100
    pct_sws = df['Gen_SWS'] / total_gen * 100
    pct_mmv = df['Gen_MMV'] / total_gen * 100
    
    plt.stackplot(df[x_col], pct_td, pct_psd, pct_sws, pct_mmv, 
                  labels=['TD', 'PSD', 'SWS', 'MIV'],
                  colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    
    plt.xlim(df[x_col].min(), df[x_col].max())
    plt.ylim(0, 100)
    plt.xlabel(x_label)
    plt.ylabel('Normalized Source Fraction [%]')
    plt.title('Generation Ratio of Source Processes (100% Stacked)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()

def plot_generation_absolute_flux(target_dir, data_source, x_axis_mode):
    """ 各ソースプロセスごとの絶対生成量（全球平均フラックス [atoms/cm^2/s]）の推移図 """
    df, x_col, x_label = get_budget_data(target_dir, data_source, x_axis_mode)
    plt.figure(figsize=(10, 6))
    
    # 物理定数
    RM_cm = 2.440e8  # 水星半径 [cm]
    global_area_cm2 = 4 * np.pi * (RM_cm ** 2)
    
    # ★ データソースに応じて「1行あたりの経過時間」を切り替える ★
    if data_source == 'FinalYearAveraged':
        # 1年間(約760万秒)を360個のTAAビンに分割して累積しているため、
        # 1つのビンには (1年 / 360) 秒分のデータが詰まっている
        mercury_year_sec = 87.969 * 86400
        time_per_row = mercury_year_sec / 360.0
    else:
        # TimeSeries は1ステップごとの生データ
        time_per_row = 100.0  # DT_MOVE [s]

    conversion_factor = global_area_cm2 * time_per_row
    
    # 単位を [atoms/cm^2/s] に変換
    flux_td = df['Gen_TD'] / conversion_factor
    flux_psd = df['Gen_PSD'] / conversion_factor
    flux_sws = df['Gen_SWS'] / conversion_factor
    flux_mmv = df['Gen_MMV'] / conversion_factor
    
    # ゼロ割や対数エラーを防ぐための微小値への置換
    BASELINE = 1e-5
    
    plt.plot(df[x_col], flux_td.replace(0, BASELINE), label='TD Flux', color='orange', linewidth=2.5, linestyle='-')
    plt.plot(df[x_col], flux_psd.replace(0, BASELINE), label='PSD Flux', color='blue', linewidth=2.5, linestyle='-')
    plt.plot(df[x_col], flux_sws.replace(0, BASELINE), label='SWS Flux', color='green', linewidth=2.5, linestyle='-')
    plt.plot(df[x_col], flux_mmv.replace(0, BASELINE), label='MIV Flux', color='purple', linewidth=2.5, linestyle='-')
    
    # 変動が激しいためY軸は対数スケール
    plt.yscale('log')
    plt.ylim(bottom=1e3)
    plt.xlim(df[x_col].min(), df[x_col].max())
    
    plt.xlabel(x_label)
    plt.ylabel('Global Avg Generation Flux [atoms/cm$^2$/s]')
    plt.title('Absolute Generation Flux by Source Process')
    
    # 凡例を枠外に配置
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_budget_absolute(target_dir, data_source, x_axis_mode):
    df, x_col, x_label = get_budget_data(target_dir, data_source, x_axis_mode)
    plt.figure(figsize=(10, 6))
    
    BASELINE = 1e24
    
    # --- Loss ---
    plt.plot(df[x_col], df['Loss_Total'].replace(0, BASELINE), color='red', linewidth=7, alpha=0.3, label='Total Loss (Background)')
    plt.plot(df[x_col], df['Loss_Stuck'].replace(0, BASELINE), label='Loss: Stuck', linestyle=':', color='darkred')
    plt.plot(df[x_col], df['Loss_Escaped'].replace(0, BASELINE), label='Loss: Escaped', linestyle=':', color='purple')
    plt.plot(df[x_col], df['Loss_Ionized'].replace(0, BASELINE), label='Loss: Ionized', linestyle=':', color='magenta')

    # --- Generation ---
    plt.plot(df[x_col], df['Gen_Total'].replace(0, BASELINE), color='black', linewidth=2, linestyle='--', label='Total Generation')
    plt.plot(df[x_col], df['Gen_TD'].replace(0, BASELINE), label='Gen: TD', linestyle='-.', color='orange')
    plt.plot(df[x_col], df['Gen_PSD'].replace(0, BASELINE), label='Gen: PSD', linestyle='-.', color='blue')
    
    plt.yscale('log')
    plt.ylim(bottom=BASELINE)
    plt.xlim(df[x_col].min(), df[x_col].max())
    
    plt.xlabel(x_label)
    plt.ylabel('Flux [atoms / step_interval]')
    plt.title('Absolute Source & Loss Budget')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_planetary_depletion(target_dir, data_source, x_axis_mode):
    df, x_col, x_label = get_budget_data(target_dir, data_source, x_axis_mode)
    plt.figure(figsize=(10, 6))
    
    cum_escape = df['Loss_Escaped'].cumsum()
    cum_ionized = df['Loss_Ionized'].cumsum()
    cum_space_loss = cum_escape + cum_ionized 
    
    if 'Supply_Internal' in df.columns:
        cum_supply = df['Supply_Internal'].cumsum()
    else:
        cum_supply = np.zeros(len(df))
        
    net_depletion = cum_space_loss - cum_supply
    
    plt.plot(df[x_col], cum_space_loss, color='red', linewidth=2.5, label='Cumulative Loss to Space (Escaped + Ionized)')
    plt.plot(df[x_col], cum_supply, color='blue', linewidth=2.5, label='Cumulative Supply (Internal Bulk Diffusion)')
    plt.plot(df[x_col], net_depletion, color='black', linewidth=3, linestyle='--', label='Net Depletion (Loss - Supply)')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5)
    
    plt.xlim(df[x_col].min(), df[x_col].max())
    
    plt.xlabel(x_label)
    plt.ylabel('Cumulative Flux [atoms]')
    plt.title('Planetary Na Depletion over Time (Surface Mass Balance)')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_bin_inventory_evolution(target_dir, data_source, x_axis_mode):
    files = glob.glob(os.path.join(target_dir, "surface_density_t*.npy"))
    if not files:
        print("[スキップ] surface_density_t*.npy が見つかりません。")
        return
    
    time_file_pairs = []
    for f in files:
        m = re.search(r'surface_density_t(\d+)\.npy', os.path.basename(f))
        if m:
            time_file_pairs.append((int(m.group(1)), f))
    time_file_pairs.sort()
    
    RM = 2.440e6
    temp_dens = np.load(time_file_pairs[0][1])
    n_lon, n_lat, n_bins = temp_dens.shape
    
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    dlon = 2 * np.pi / n_lon
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    total_area = 4 * np.pi * (RM ** 2)
    
    times = []
    bin_inventories = []
    
    for t, f in time_file_pairs:
        surf_dens = np.load(f)
        weighted_dens = surf_dens * cell_areas[np.newaxis, :, np.newaxis]
        total_atoms_per_bin = np.sum(weighted_dens, axis=(0, 1))
        mean_dens_cm2 = (total_atoms_per_bin / total_area) / 1e4
        times.append(t)
        bin_inventories.append(mean_dens_cm2)
        
    times = np.array(times)
    bin_inventories = np.array(bin_inventories)
    
    # 3年分のnpyファイルの横軸マッピング
    if data_source == 'TimeSeries' and x_axis_mode == 'TAA':
        ts_csv_path = os.path.join(target_dir, "budget_timeseries.csv")
        if os.path.exists(ts_csv_path):
            df_ts = pd.read_csv(ts_csv_path)
            diff = df_ts['TAA'].diff()
            wrap_count = (diff < -180).cumsum().fillna(0)
            df_ts['Unwrapped_TAA'] = df_ts['TAA'] + wrap_count * 360.0
            
            x_data = []
            for t_val in times:
                idx = (np.abs(df_ts['Time_hours'] - t_val)).argmin()
                x_data.append(df_ts['Unwrapped_TAA'].iloc[idx])
            x_data = np.array(x_data)
            x_label = 'Cumulative TAA [deg]'
        else:
            x_data = times
            x_label = 'Simulation Time [hours]'
    else:
        x_data = times
        x_label = 'Simulation Time [hours]'
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n_bins))
    for b in range(n_bins):
        if b == 0:
            label_str = 'Shallowest Bin 0 (Easily Desorbed)'
        elif b == n_bins - 1:
            label_str = f'Deepest Bin {b} (Subsurface Reservoir)'
        else:
            label_str = f'Intermediate Bin {b}'
            
        plt.plot(x_data, bin_inventories[:, b], label=label_str, color=colors[b], linewidth=2.5)
        
    plt.xlabel(x_label)
    plt.ylabel('Global Average Density [atoms/cm$^2$]')
    plt.title('Evolution of Global Average Na Density per Bin')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def _load_bin_inventory_data(target_dir, data_source, x_axis_mode):
    """追加関数用の共通データ読み込みヘルパー（FinalYearAveragedでの3年目抽出に対応）"""
    files = glob.glob(os.path.join(target_dir, "surface_density_t*.npy"))
    if not files:
        return None, None, None, None
    
    time_file_pairs = []
    for f in files:
        m = re.search(r'surface_density_t(\d+)\.npy', os.path.basename(f))
        if m:
            time_file_pairs.append((int(m.group(1)), f))
    time_file_pairs.sort()
    
    RM = 2.440e6
    temp_dens = np.load(time_file_pairs[0][1])
    n_lon, n_lat, n_bins = temp_dens.shape
    
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    dlon = 2 * np.pi / n_lon
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    total_area = 4 * np.pi * (RM ** 2)
    
    times = []
    bin_inventories = []
    
    for t, f in time_file_pairs:
        surf_dens = np.load(f)
        weighted_dens = surf_dens * cell_areas[np.newaxis, :, np.newaxis]
        total_atoms_per_bin = np.sum(weighted_dens, axis=(0, 1))
        mean_dens_cm2 = (total_atoms_per_bin / total_area) / 1e4
        times.append(t)
        bin_inventories.append(mean_dens_cm2)
        
    times = np.array(times)
    bin_inventories = np.array(bin_inventories)
    
    ts_csv_path = os.path.join(target_dir, "budget_timeseries.csv")
    if os.path.exists(ts_csv_path):
        df_ts = pd.read_csv(ts_csv_path)
        diff = df_ts['TAA'].diff()
        wrap_count = (diff < -180).cumsum().fillna(0)
        df_ts['Unwrapped_TAA'] = df_ts['TAA'] + wrap_count * 360.0
        
        x_data_uw = []
        x_data_taa = []
        for t_val in times:
            idx = (np.abs(df_ts['Time_hours'] - t_val)).argmin()
            x_data_uw.append(df_ts['Unwrapped_TAA'].iloc[idx])
            x_data_taa.append(df_ts['TAA'].iloc[idx])
        
        x_data_uw = np.array(x_data_uw)
        x_data_taa = np.array(x_data_taa)
        
        if data_source == 'FinalYearAveraged':
            # 最後の360度分（最終年）のみを抽出し、X軸を0-360のTAAにする
            threshold = max(0, x_data_uw.max() - 360.0)
            mask = x_data_uw >= (threshold - 1e-5)
            x_data = x_data_taa[mask]
            bin_inventories = bin_inventories[mask]
            x_label = 'TAA (Final Year) [deg]'  # 変更: Final Year とする
        elif data_source == 'TimeSeries' and x_axis_mode == 'TAA':
            x_data = x_data_uw
            x_label = 'Cumulative TAA [deg]'
        else:
            x_data = times
            x_label = 'Simulation Time [hours]'
    else:
        # CSVがない場合のフォールバック
        if data_source == 'FinalYearAveraged':
            # 水星の1年（約2111時間）分を最大時間から遡って抽出する
            mercury_year_hours = 87.969 * 24
            threshold_time = max(0, times[-1] - mercury_year_hours)
            mask = times >= threshold_time
            x_data = times[mask]
            bin_inventories = bin_inventories[mask]
            x_label = 'Simulation Time [hours] (Final Year)' # 変更: Final Year とする
        else:
            x_data = times
            x_label = 'Simulation Time [hours]'
            
    return x_data, bin_inventories, n_bins, x_label

def plot_total_surface_atoms(target_dir, data_source, x_axis_mode):
    """
    表面密度から全表面の総原子数を計算し、TAA（または時間）に対してプロットする
    """
    x_data, bin_inventories, n_bins, x_label = _load_bin_inventory_data(target_dir, data_source, x_axis_mode)
    if x_data is None or len(x_data) == 0:
        print("[スキップ] 表面原子数グラフ用のデータが見つかりません。")
        return
        
    RM_cm = 2.440e8
    total_area_cm2 = 4 * np.pi * (RM_cm ** 2)
    total_atoms_series = np.sum(bin_inventories, axis=1) * total_area_cm2
    
    # 値が急激に減少する箇所（例: 360度から0度に戻る場所）を見つける
    diffs = np.diff(x_data)
    wrap_indices = np.where(diffs < -180)[0] + 1  # -180度以上の急減を検知
    
    if len(wrap_indices) > 0:
        # 巻き戻る箇所に np.nan (欠損値) を挿入して、matplotlibに線を切らせる
        x_data_plot = np.insert(x_data.astype(float), wrap_indices, np.nan)
        total_atoms_plot = np.insert(total_atoms_series.astype(float), wrap_indices, np.nan)
    else:
        x_data_plot = x_data.astype(float)
        total_atoms_plot = total_atoms_series.astype(float)

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(x_data_plot, total_atoms_plot, color='teal', linewidth=2.5, label='Total Surface Na Atoms')
    
    plt.xlabel(x_label)
    plt.ylabel('Total Number of Atoms')
    plt.title('Evolution of Total Surface Na Atoms')
    
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(60))
    if 'TAA' in x_label or 'deg' in x_label:
        plt.xlim(0, 360)

    plt.legend(loc='upper right')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_td_flux_model_comparison(target_dir, data_source, x_axis_mode):
    """
    【追加機能】表面密度の推移を用いて、1次反応と2次反応での
    熱脱離フラックスの「密度依存項（減衰の激しさ）」の違いを比較するプロット。
    """
    x_data, bin_inventories, n_bins, x_label = _load_bin_inventory_data(target_dir, data_source, x_axis_mode)
    if x_data is None or len(x_data) == 0:
        print("[スキップ] 比較用のデータが見つかりません。")
        return

    # 全表面の平均密度 (全ビン合計)
    total_density = np.sum(bin_inventories, axis=1)

    # ==========================================
    # 2つのモデルの「密度依存項」を計算
    # ※実際のフラックスはこれに温度項( exp(-E/kT) )が掛かりますが、
    # 今回は「密度低下による出にくさ」のカーブ形状を比較します。
    # ==========================================
    
    # 1次反応モデル (Leblanc 2003等): フラックス ∝ 密度
    flux_1st_order = total_density

    # 2次反応モデル (Sarantos 2021等): フラックス ∝ 密度^2
    flux_2nd_order = total_density ** 2

    # 形状（減衰の激しさ）を比較するために、それぞれの最大値で規格化（0〜1のスケールにする）
    flux_1st_norm = flux_1st_order / np.max(flux_1st_order)
    flux_2nd_norm = flux_2nd_order / np.max(flux_2nd_order)

    # グラフ描画時のラップアラウンド（360度から0度に戻る際の線の分断）処理
    diffs = np.diff(x_data)
    wrap_indices = np.where(diffs < -180)[0] + 1

    if len(wrap_indices) > 0:
        x_plot = np.insert(x_data.astype(float), wrap_indices, np.nan)
        f1_plot = np.insert(flux_1st_norm.astype(float), wrap_indices, np.nan)
        f2_plot = np.insert(flux_2nd_norm.astype(float), wrap_indices, np.nan)
    else:
        x_plot = x_data.astype(float)
        f1_plot = flux_1st_norm.astype(float)
        f2_plot = flux_2nd_norm.astype(float)

    # グラフの描画
    plt.figure(figsize=(10, 6))
    
    # 1次反応のプロット
    plt.plot(x_plot, f1_plot, color='blue', linewidth=2.5, linestyle='-', 
             label='1st-Order (Leblanc 2003): $\propto N$')
    
    # 2次反応のプロット
    plt.plot(x_plot, f2_plot, color='red', linewidth=3, linestyle='--', 
             label='2nd-Order (Sarantos 2021): $\propto N^2$')

    plt.xlabel(x_label)
    plt.ylabel('Normalized Density-Dependent Factor')
    plt.title('Comparison of TD Flux Suppresion by Density\n(1st-Order vs 2nd-Order)')
    
    # X軸の目盛り調整
    from matplotlib.ticker import MultipleLocator
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(60))
    if 'TAA' in x_label or 'deg' in x_label:
        plt.xlim(0, 360)

    plt.legend(loc='upper right')
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_bin_inventory_interactive_histogram(target_dir, data_source, x_axis_mode):
    """【修正】横軸をeV表記にしたスライダー＆矢印キー対応インタラクティブ図"""
    x_data, bin_inventories, n_bins, x_label = _load_bin_inventory_data(target_dir, data_source, x_axis_mode)
    if x_data is None or len(x_data) == 0:
        print("[スキップ] グラフ描画用のデータが見つかりません。")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)
    
    # ★ 1.4eV 〜 2.7eV を n_bins 等分してビン幅と中心値を計算 ★
    bin_edges = np.linspace(1.4, 2.7, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (2.7 - 1.4) / n_bins * 0.8  # 0.8は棒の間に少し隙間を作るため
    
    max_y = np.max(bin_inventories) * 1.1
    
    bars = ax.bar(bin_centers, bin_inventories[0], width=bin_width, color='skyblue', edgecolor='black')
    
    ax.set_xlabel('Activation Energy [eV]')
    ax.set_ylabel('Global Average Density [atoms/cm$^2$]')
    ax.set_title(f'Bin Inventory Distribution ( {x_label}: {x_data[0]:.2f} )')
    ax.set_ylim(0, max_y)
    
    # X軸のメモリをビンの中心値(eV)に設定し、小数点以下2桁で表示
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{val:.2f}" for val in bin_centers], rotation=45)
    ax.grid(axis='y', ls="--", alpha=0.5)
    
    # スライダーのUI設定
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax=ax_slider, label='Step Index', valmin=0, valmax=len(x_data) - 1, valinit=0, valstep=1, valfmt='%d')
    
    def update(val):
        idx = int(slider.val)
        for i, bar in enumerate(bars):
            bar.set_height(bin_inventories[idx][i])
        ax.set_title(f'Bin Inventory Distribution ( {x_label}: {x_data[idx]:.2f} )')
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    
    # 矢印キーで動かすイベント
    def on_key(event):
        if event.key in ['right', 'up']:
            new_val = min(slider.val + 1, slider.valmax)
            slider.set_val(new_val)
        elif event.key in ['left', 'down']:
            new_val = max(slider.val - 1, slider.valmin)
            slider.set_val(new_val)
            
    fig.canvas.mpl_connect('key_press_event', on_key)
    plot_bin_inventory_interactive_histogram.slider = slider
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) # tight_layoutで潰れたスライダーの余白を再確保
    plt.show()

def export_bin_inventory_gif(target_dir, data_source, x_axis_mode, output_filename="bin_evolution.gif"):
    """【追加】最初から最後まで推移するGIFアニメーションを出力する関数"""
    x_data, bin_inventories, n_bins, x_label = _load_bin_inventory_data(target_dir, data_source, x_axis_mode)
    if x_data is None or len(x_data) == 0:
        print("[スキップ] GIF生成用のデータが見つかりません。")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bin_edges = np.linspace(1.4, 2.7, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (2.7 - 1.4) / n_bins * 0.8
    
    max_y = np.max(bin_inventories) * 1.1
    bars = ax.bar(bin_centers, bin_inventories[0], width=bin_width, color='skyblue', edgecolor='black')
    
    ax.set_xlabel('Activation Energy [eV]')
    ax.set_ylabel('Global Average Density [atoms/cm$^2$]')
    ax.set_ylim(0, max_y)
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{val:.2f}" for val in bin_centers], rotation=45)
    ax.grid(axis='y', ls="--", alpha=0.5)
    
    title_text = ax.set_title(f'Bin Inventory Distribution ( {x_label}: {x_data[0]:.2f} )')
    plt.tight_layout()
    
    def animate(i):
        for j, bar in enumerate(bars):
            bar.set_height(bin_inventories[i][j])
        title_text.set_text(f'Bin Inventory Distribution ( {x_label}: {x_data[i]:.2f} )')
        return bars
        
    # fps=10 は1秒間に10フレーム進む設定（調整可能）
    anim = FuncAnimation(fig, animate, frames=len(x_data), interval=100, blit=False)
    
    output_path = os.path.join(target_dir, output_filename)
    print(f"GIFアニメーションを作成中... (フレーム数: {len(x_data)}) 少し時間がかかります。")
    
    # 保存処理
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"✅ GIFの保存が完了しました: {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    print(f"=== 結果ディレクトリの解析開始: {RESULT_DIR} ===")
    print(f"=== データソース: {DATA_SOURCE} | X軸: {X_AXIS_MODE} ===")
    
    if not os.path.exists(RESULT_DIR):
        print("エラー: 指定されたディレクトリが存在しません。パスを修正してください。")
    else:
        try:
            plot_generation_ratio(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            plot_generation_absolute_flux(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)   
            plot_budget_absolute(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            plot_planetary_depletion(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            plot_bin_inventory_evolution(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            plot_total_surface_atoms(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            plot_bin_inventory_interactive_histogram(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            plot_td_flux_model_comparison(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            export_bin_inventory_gif(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE, output_filename="bin_evolution_animation.gif")

        except Exception as e:
            print(f"エラーが発生しました: {e}")
            
    print("=== 解析完了 ===")