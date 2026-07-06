# -*- coding: utf-8 -*-
"""
dt=1s と dt=100s のビンインベントリ比較スクリプト (最終年のみ抽出)
- 2次元ヒートマップによる全球平均密度の比較 (横軸: TAA, 縦軸: 束縛エネルギー)
- 特定のTAA (近日点・遠日点など) における分布形状（ピークシフト）の直接比較
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re

# ==========================================
# 1. 設定
# ==========================================
# 比較する2つのディレクトリパスを指定してください
DIR_DT1 = r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT1_0616_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)" # dt=1sのパス
DIR_DT100 = r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0609_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1e24" # dt=100sのパス

# ==========================================
# 2. データ読み込みヘルパー関数
# ==========================================
def load_final_year_bins(target_dir):
    """
    指定ディレクトリからsurface_densityのnpyファイルを読み込み、
    最終年（3年目）のTAAとビンごとの全球平均密度を返す。
    """
    files = glob.glob(os.path.join(target_dir, "surface_density_t*.npy"))
    if not files:
        print(f"警告: データが見つかりません -> {target_dir}")
        return None, None, None

    # 時間順にソート
    time_file_pairs = []
    for f in files:
        m = re.search(r'surface_density_t(\d+)\.npy', os.path.basename(f))
        if m:
            time_file_pairs.append((int(m.group(1)), f))
    time_file_pairs.sort()

    # 水星の表面積とセルの面積計算
    RM = 2.440e6
    temp_dens = np.load(time_file_pairs[0][1])
    n_lon, n_lat, n_bins = temp_dens.shape
    lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
    dlon = 2 * np.pi / n_lon
    cell_areas = (RM ** 2) * dlon * (np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    total_area = 4 * np.pi * (RM ** 2)

    times = []
    bin_inventories = []

    # 全ファイルの密度を読み込み、全球平均 [atoms/cm^2] を計算
    for t, f in time_file_pairs:
        surf_dens = np.load(f)
        weighted_dens = surf_dens * cell_areas[np.newaxis, :, np.newaxis]
        total_atoms_per_bin = np.sum(weighted_dens, axis=(0, 1))
        mean_dens_cm2 = (total_atoms_per_bin / total_area) / 1e4
        times.append(t)
        bin_inventories.append(mean_dens_cm2)

    times = np.array(times)
    bin_inventories = np.array(bin_inventories)

    # budget_timeseries.csv を使って時間 -> TAA へのマッピングと最終年の抽出
    ts_csv_path = os.path.join(target_dir, "budget_timeseries.csv")
    if not os.path.exists(ts_csv_path):
        print(f"警告: 時系列CSVが見つかりません -> {ts_csv_path}")
        return None, None, None

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

    # 最終年 (最後の360度分) のみを抽出
    threshold = max(0, x_data_uw.max() - 360.0)
    mask = x_data_uw >= (threshold - 1e-5)
    
    final_taa = x_data_taa[mask]
    final_bins = bin_inventories[mask]

    # TAA順にソート (0 -> 360できれいにプロットするため)
    sort_idx = np.argsort(final_taa)
    final_taa = final_taa[sort_idx]
    final_bins = final_bins[sort_idx]

    # ビンのエネルギー中心値を計算 (1.4eV ~ 2.7eV)
    bin_edges = np.linspace(1.4, 2.7, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return final_taa, final_bins, bin_centers

# ==========================================
# 3. グラフ描画関数群
# ==========================================

def plot_heatmap_comparison(taa1, bins1, taa100, bins100, bin_centers):
    """ TAA vs 束縛エネルギーのヒートマップ比較 """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # カラーマップのスケールを両者で統一するための最大値取得
    vmax = max(np.max(bins1), np.max(bins100))
    vmin = 0

    # extent = [x_min, x_max, y_min, y_max]
    extent = [0, 360, bin_centers[-1], bin_centers[0]] # imshowはy軸が逆になりがちなので調整

    # dt=1s
    im1 = ax1.imshow(bins1.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                     extent=[0, 360, 2.7, 1.4], origin='upper')
    ax1.set_title('dt = 1s (Reference)')
    ax1.set_xlabel('TAA [deg]')
    ax1.set_ylabel('Activation Energy [eV]')

    # dt=100s
    im2 = ax2.imshow(bins100.T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax,
                     extent=[0, 360, 2.7, 1.4], origin='upper')
    ax2.set_title('dt = 100s')
    ax2.set_xlabel('TAA [deg]')

    fig.colorbar(im2, ax=[ax1, ax2], label='Global Average Density [atoms/cm$^2$]')
    fig.suptitle('Bin Inventory Evolution over Final Year (Heatmap)', fontsize=16)
    
    # 軸の向きを修正 (下を1.4eV、上を2.7eVに)
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    
    plt.show()


def plot_snapshot_comparison(taa1, bins1, taa100, bins100, bin_centers, target_taas=[0, 180]):
    """ 指定したTAAにおけるビン分布の形状（ピークシフト等）を比較 """
    n_plots = len(target_taas)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), sharey=True)
    if n_plots == 1:
        axes = [axes]

    bin_width = (bin_centers[1] - bin_centers[0]) * 0.35 # 棒を2本並べるための幅

    for i, t_taa in enumerate(target_taas):
        ax = axes[i]
        
        # 指定TAAに最も近いインデックスを取得
        idx1 = (np.abs(taa1 - t_taa)).argmin()
        idx100 = (np.abs(taa100 - t_taa)).argmin()

        # dt=1s は青、dt=100s は赤でプロット
        ax.bar(bin_centers - bin_width/2, bins1[idx1], width=bin_width, 
               label='dt = 1s', color='royalblue', edgecolor='black', alpha=0.8)
        ax.bar(bin_centers + bin_width/2, bins100[idx100], width=bin_width, 
               label='dt = 100s', color='crimson', edgecolor='black', alpha=0.8)

        ax.set_title(f'TAA $\\approx$ {t_taa}°\n(dt1: {taa1[idx1]:.1f}°, dt100: {taa100[idx100]:.1f}°)')
        ax.set_xlabel('Activation Energy [eV]')
        if i == 0:
            ax.set_ylabel('Global Average Density [atoms/cm$^2$]')
        
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([f"{val:.2f}" for val in bin_centers], rotation=45)
        ax.grid(axis='y', ls="--", alpha=0.5)
        ax.legend()

    fig.suptitle('Binding Energy Distribution Comparison', fontsize=16)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. 実行ブロック
# ==========================================
if __name__ == "__main__":
    print("=== データ読み込み開始 ===")
    taa1, bins1, centers = load_final_year_bins(DIR_DT1)
    taa100, bins100, centers_100 = load_final_year_bins(DIR_DT100)

    if taa1 is not None and taa100 is not None:
        print("=== グラフ生成中 ===")
        # 1. ヒートマップ比較 (3年目の全体の推移を俯瞰)
        plot_heatmap_comparison(taa1, bins1, taa100, bins100, centers)
        
        # 2. スナップショット比較 (近日点=0度 と 遠日点=180度 での分布形状の違いを確認)
        # ※ピークが深いビンにシフトしているか、蓄積量がどれくらい違うかが見えます
        plot_snapshot_comparison(taa1, bins1, taa100, bins100, centers, target_taas=[0, 180])
        print("=== 完了 ===")
    else:
        print("データの読み込みに失敗したため、処理を終了します。パスを確認してください。")