# -*- coding: utf-8 -*-
"""
水星ナトリウムシミュレーション
年度ごとの Total, Surface, Exosphere 在庫推移と差分（収束確認）を解析するスクリプト
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from matplotlib.ticker import MultipleLocator

# ==========================================
# 1. 設定
# ==========================================
RESULT_DIR = r"./SimulationResult_202607/ParabolicHop_72x36_NoEq_DT100_0731_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_25yr_FULL"

MERCURY_ECCENTRICITY = 0.205630

# ==========================================
# 2. 解析メイン関数
# ==========================================
def plot_yearly_evolution(target_dir):
    csv_path = os.path.join(target_dir, "budget_timeseries.csv")
    if not os.path.exists(csv_path):
        print(f"エラー: CSVファイルが見つかりません: {csv_path}")
        return

    print("データを読み込み中...")
    df = pd.read_csv(csv_path)
    
    if 'Surface_Total' not in df.columns or 'Exosphere_Total' not in df.columns:
        print("エラー: 必要なデータ(Surface_Total, Exosphere_Total)がCSVにありません。")
        return

    # --- データの準備 (年数の計算) ---
    # TAAの巻き戻り(-180度以下の急減)を利用して累積TAAを計算
    diff = df['TAA'].diff()
    wrap_count = (diff < -180).cumsum().fillna(0)
    df['Unwrapped_TAA'] = df['TAA'] + wrap_count * 360.0
    
    # 最初の周回を Year 1 とする
    df['Year'] = (df['Unwrapped_TAA'] // 360.0).astype(int) + 1
    
    # 系全体のトータル
    df['System_Total'] = df['Surface_Total'] + df['Exosphere_Total']

    years = sorted(df['Year'].unique())
    
    # ケプラーの第2法則に基づく時間重み付け (Time-Weighted Average用)
    def get_weights(taa):
        taa_rad = np.deg2rad(taa)
        return 1.0 / (1.0 + MERCURY_ECCENTRICITY * np.cos(taa_rad))**2

    avg_surface, avg_exo, avg_system, year_labels = [], [], [], []

    # ==========================================
    # グラフ1: TAAごとの波形を年ごとに重ね書き
    # ==========================================
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors = cm.viridis(np.linspace(0, 1, len(years)))

    print("\n=== 年ごとのTWA (Time-Weighted Average) 在庫推移と前年比 ===")
    
    for i, yr in enumerate(years):
        df_yr = df[df['Year'] == yr]
        
        # 1年未満の不完全なデータ（シミュ開始・終了直後）は計算から除外
        if len(df_yr) < 50: 
            continue
            
        # 描画
        ax1.plot(df_yr['TAA'], df_yr['System_Total'], color=colors[i], lw=1.5)
        ax2.plot(df_yr['TAA'], df_yr['Surface_Total'], color=colors[i], lw=1.5)
        ax3.plot(df_yr['TAA'], df_yr['Exosphere_Total'], color=colors[i], lw=1.5)
        
        # TWAの計算
        w = get_weights(df_yr['TAA'])
        a_sys = np.average(df_yr['System_Total'], weights=w)
        a_surf = np.average(df_yr['Surface_Total'], weights=w)
        a_exo = np.average(df_yr['Exosphere_Total'], weights=w)
        
        avg_system.append(a_sys)
        avg_surface.append(a_surf)
        avg_exo.append(a_exo)
        year_labels.append(yr)
        
        # --- コンソール出力 (収束確認) ---
        print(f"Year {yr}:")
        print(f"  System Total : {a_sys:.4e} [atoms]")
        print(f"  Surface      : {a_surf:.4e} [atoms]")
        print(f"  Exosphere    : {a_exo:.4e} [atoms]")
        
        if len(year_labels) > 1:
            prev_sys = avg_system[-2]
            prev_surf = avg_surface[-2]
            prev_exo = avg_exo[-2]
            diff_sys = (a_sys - prev_sys) / prev_sys * 100.0 if prev_sys > 0 else 0
            diff_surf = (a_surf - prev_surf) / prev_surf * 100.0 if prev_surf > 0 else 0
            diff_exo = (a_exo - prev_exo) / prev_exo * 100.0 if prev_exo > 0 else 0
            print(f"  => Change vs Last Year: Sys={diff_sys:+.3f}%, Surf={diff_surf:+.3f}%, Exo={diff_exo:+.3f}%")
        print("-" * 60)

    # グラフ1の装飾
    ax1.set_title('Yearly Evolution of System Total (Surf + Exo)')
    ax1.set_ylabel('System Total [atoms]')
    ax2.set_title('Yearly Evolution of Surface Total')
    ax2.set_ylabel('Surface Total [atoms]')
    ax3.set_title('Yearly Evolution of Exosphere Total')
    ax3.set_ylabel('Exosphere Total [atoms]')
    ax3.set_xlabel('True Anomaly Angle (TAA) [deg]')
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, ls='--', alpha=0.5)
        ax.yaxis.get_major_formatter().set_scientific(True)
        ax.yaxis.get_major_formatter().set_useOffset(False)
        ax.xaxis.set_major_locator(MultipleLocator(60))

    ax3.set_xlim(0, 360)
    
    # カラーバーの追加 (年数の凡例)
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=min(years), vmax=max(years)))
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.08, pad=0.02)
    cbar.set_label('Simulation Year')

    plt.tight_layout()
    plt.show()

    # ==========================================
    # グラフ2: TWA（年平均）の経年推移プロット
    # ==========================================
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # 系全体と表面はスケールが近いので左軸
    ax.plot(year_labels, avg_system, marker='o', color='black', lw=2.5, label='System Total (Surf+Exo)')
    ax.plot(year_labels, avg_surface, marker='s', color='teal', lw=2.5, label='Surface Total')
    
    # 大気(Exo)は桁が違うので右軸に設定
    ax_sub = ax.twinx()
    ax_sub.plot(year_labels, avg_exo, marker='^', color='blue', lw=2.5, label='Exosphere Total')
    
    ax.set_title('Year-over-Year Trend of Time-Weighted Average Na Inventory')
    ax.set_xlabel('Simulation Year')
    
    ax.set_ylabel('System & Surface Stock [atoms]', fontweight='bold')
    ax_sub.set_ylabel('Exosphere Stock [atoms]', color='blue', fontweight='bold')
    ax_sub.tick_params(axis='y', labelcolor='blue')
    
    # 左右軸それぞれの指数表記を見やすく
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax_sub.yaxis.get_major_formatter().set_scientific(True)
    ax_sub.yaxis.get_major_formatter().set_useOffset(False)
    
    ax.grid(True, ls='--', alpha=0.5)
    
    # 凡例をまとめる
    lines = ax.get_lines() + ax_sub.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    # X軸を整数（年）に
    ax.xaxis.set_major_locator(MultipleLocator(1))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print(f"=== 年次推移解析 実行開始 ===")
    plot_yearly_evolution(RESULT_DIR)
    print("=== 完了 ===")