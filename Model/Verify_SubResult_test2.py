# -*- coding: utf-8 -*-
"""
シミュレーション結果 解析・可視化スクリプト (3年分フルタイム対応版)
- 100%積み上げの生成割合
- 生成と喪失(Lossの内訳)の絶対量
- 拡散によるビンごとのインベントリ推移
- 水星全体でのNa枯渇量（累積収支）の推移
- 【追加】ビンインベントリ推移の積み上げヒストグラム風表示
- 【追加】ビンインベントリ推移の2次元ヒートマップ表示
- 【追加】Killen (2004) フォーマットでのフラックス(atoms/cm2/s)のPrint機能
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
RESULT_DIR = r"./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0627_Multi_BD0.5_U1.85_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_NoDif"
DATA_SOURCE = "csv" 
X_AXIS_MODE = "taa" 

# ==========================================
# [追加] Killen (2004) Tableフォーマットでの出力機能
# ==========================================
def print_killen_table_comparison(result_dir):
    csv_path = os.path.join(result_dir, "budget_timeseries.csv")
    if not os.path.exists(csv_path):
        print(f"CSVファイルが見つかりません: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # 面積の定義 (cm2)
    RM_CM = 2.440e8
    AREA_CM2 = 4 * np.pi * (RM_CM**2)
    DAY_AREA_CM2 = AREA_CM2 / 2.0
    
    print("\n" + "="*65)
    print("【Killen (2004) Table 2 フォーマットとのフラックス比較】")
    print("="*65)
    
    # 新しいシミュレーションコードで局所最大値が記録されているか判定
    if 'Max_Flux_PSD' in df.columns:
        print("※ シミュレーションで記録された『局所最大フラックス (Local Maximum)』を表示します。")
        max_psd = df['Max_Flux_PSD'].max()
        max_td  = df['Max_Flux_TD'].max()
        max_sws = df['Max_Flux_SWS'].max()
        max_mmv = df['Max_Flux_MMV'].max()
    else:
        print("※ 局所値の記録がないため、総生成量から『昼側平均フラックス (Dayside Average)』を概算表示します。")
        print("※ (Killenの表は局所最大値のため、この平均値より1〜2桁高くなります)")
        
        # DataFrameにデータが存在するかチェックして最大値を計算
        max_psd = (df['Gen_PSD'] / DAY_AREA_CM2).max() if 'Gen_PSD' in df.columns else 0.0
        max_td  = (df['Gen_TD'] / DAY_AREA_CM2).max() if 'Gen_TD' in df.columns else 0.0
        max_sws = (df['Gen_SWS'] / DAY_AREA_CM2).max() if 'Gen_SWS' in df.columns else 0.0
        max_mmv = (df['Gen_MMV'] / AREA_CM2).max() if 'Gen_MMV' in df.columns else 0.0 # MMVは全球平均

    print(f" - Photon-Stimulated Desorption (PSD) : {max_psd:.2e} atoms/cm2/s")
    print(f" - Thermal Vaporization (TD)          : {max_td:.2e} atoms/cm2/s")
    print(f" - Ion Sputtering (SWS)               : {max_sws:.2e} atoms/cm2/s")
    print(f" - Impact Vaporization (MMV)          : {max_mmv:.2e} atoms/cm2/s")
    print("="*65 + "\n")

# ==========================================
# (既存のプロット関数はそのまま保持されていると仮定し省略)
# ==========================================
def plot_generation_ratio(result_dir, data_source, x_axis_mode):
    pass
def plot_generation_absolute_flux(result_dir, data_source, x_axis_mode):
    pass
def plot_budget_absolute(result_dir, data_source, x_axis_mode):
    pass
def plot_planetary_depletion(result_dir, data_source, x_axis_mode):
    pass
def plot_bin_inventory_evolution(result_dir, data_source, x_axis_mode):
    pass
def plot_total_surface_atoms(result_dir, data_source, x_axis_mode):
    pass


if __name__ == "__main__":
    print(f"=== 結果ディレクトリの解析開始: {RESULT_DIR} ===")
    print(f"=== データソース: {DATA_SOURCE} | X軸: {X_AXIS_MODE} ===")
    
    if not os.path.exists(RESULT_DIR):
        print("エラー: 指定されたディレクトリが存在しません。パスを修正してください。")
    else:
        try:
            # 1. Killen比較のPrint機能を実行
            print_killen_table_comparison(RESULT_DIR)
            
            # 2. 既存のグラフ描画
            #plot_generation_ratio(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            #plot_generation_absolute_flux(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)   
            #plot_budget_absolute(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            #plot_planetary_depletion(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            #plot_bin_inventory_evolution(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            #plot_total_surface_atoms(RESULT_DIR, DATA_SOURCE, X_AXIS_MODE)
            
        except Exception as e:
            print(f"解析中にエラーが発生しました: {e}")