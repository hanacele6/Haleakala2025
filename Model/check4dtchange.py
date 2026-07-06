import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 設定：タイムシリーズCSVへのパス
# ==========================================
ts_file_paths = {
    'dt = 100s': (r'./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT100_0609_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)_1e24/budget_timeseries.csv', 100.0),
    'dt = 50s':  (r'./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT50_0613_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)/budget_timeseries.csv', 50.0),
    'dt = 10s':  (r'./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT10_0614_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)/budget_timeseries.csv', 10.0),
    'dt = 1s':   (r'./SimulationResult_202606/ParabolicHop_72x36_NoEq_DT1_0616_Multi_BD0.5_UG_Q2.0_Bouncetau30s_A2.0_LongLT(Fulle)/budget_timeseries.csv', 1.0),
}

# ==========================================
# 2. データの読み込み・規格化・平滑化
# ==========================================
dfs = {}

for label, (path, dt_val) in ts_file_paths.items():
    if not os.path.exists(path):
        continue
    
    df = pd.read_csv(path).sort_values('TAA').drop_duplicates('TAA')
    
    # dtで割って秒間レート (particles/s) に規格化
    df['Gen_Rate_per_sec'] = df['Gen_Total'] / dt_val
    
    # 移動平均でノコギリ波を除去（端の欠損を防ぐために min_periods=1）
    df['Gen_Rate_Smoothed'] = df['Gen_Rate_per_sec'].rolling(window=10, center=True, min_periods=1).mean()
    
    dfs[label] = df

# ==========================================
# 3. 視覚的確認（プロット）
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# (1) 全体波形の比較（補間なしの生プロット）
for label, df in dfs.items():
    # 端の暴れを防ぐため、TAAが2度〜358度の安全な範囲だけプロット
    safe_df = df[(df['TAA'] >= 2) & (df['TAA'] <= 358)]
    ax1.plot(safe_df['TAA'], safe_df['Gen_Rate_Smoothed'], label=label, lw=2)

ax1.set_title('Smoothed Particle Generation Rate (particles/s) - Safe Bounds')
ax1.set_xlabel('True Anomaly Angle (TAA) [deg]')
ax1.set_ylabel('Generation Rate [particles/s]')
ax1.grid(True, ls='--')
ax1.legend()

# (2) 遠日点付近を強拡大してズレを可視化
for label, df in dfs.items():
    zoom_df = df[(df['TAA'] >= 170) & (df['TAA'] <= 190)]
    ax2.plot(zoom_df['TAA'], zoom_df['Gen_Rate_Smoothed'], label=label, lw=2, marker='o', markersize=3)

ax2.set_xlim(170, 190)
ax2.set_title('Zoom-in around Aphelion (TAA: 170-190 deg) - Visualizing Phase Shift')
ax2.set_xlabel('TAA [deg]')
ax2.set_ylabel('Generation Rate [particles/s]')
ax2.grid(True, which='both', ls='--')
ax2.legend()

plt.tight_layout()
plt.show()