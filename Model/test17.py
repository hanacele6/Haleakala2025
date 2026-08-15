# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401

# ==========================================
# 設定（フォルダ名はお手元の環境に合わせてください）
# ==========================================
BASE_DIR = r"./SimulationResult_202607"
MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test2",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test2",
}

def analyze_budget_comparison(models):
    data_dict = {}
    
    # データの読み込み
    for label, subdir in models.items():
        csv_path = os.path.join(BASE_DIR, subdir, "budget_statistics_per_taa.csv")
        if not os.path.exists(csv_path):
            print(f"[エラー] CSVファイルが見つかりません: {csv_path}")
            return
        data_dict[label] = pd.read_csv(csv_path)
    
    m1, m2 = list(models.keys())[0], list(models.keys())[1]  # Q2.0 と Q0.3
    df1 = data_dict[m1]
    df2 = data_dict[m2]
    
    # --- 1. Dusk側への総放出フラックスの計算 ---
    # Dusk側の大気密度を決めるのは、Dusk側で放出された「PSD + TD (+ SWS + MMV)」の合計個数です
    df1['Gen_Dusk_Total'] = df1['Gen_PSD_Dusk'] + df1['Gen_TD_Dusk']
    df2['Gen_Dusk_Total'] = df2['Gen_PSD_Dusk'] + df2['Gen_TD_Dusk']
    
    # --- 2. 数値での定量比較プリント ---
    print("="*60)
    print(" 📊 夕方側 (Dusk) へのナトリウム総放出量の定量比較")
    print("="*60)
    for taa in [120, 140, 160, 180]:
        val1 = df1.loc[df1['TAA_Bin'] == taa, 'Gen_Dusk_Total'].values[0]
        val2 = df2.loc[df2['TAA_Bin'] == taa, 'Gen_Dusk_Total'].values[0]
        ratio = val2 / val1 if val1 > 0 else np.nan
        
        print(f"📍 TAA = {taa}°:")
        print(f"  - {m1} のDusk放出量: {val1:.3e} atoms/step")
        print(f"  - {m2} のDusk放出量: {val2:.3e} atoms/step")
        print(f"  - 放出量の比率 ({m2} / {m1}): {ratio:.3f} (← 表面密度が3倍違うのに、ここは1に非常に近くなるはず)")
        print("-" * 50)

    # --- 3. プロットの作成 ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左グラフ: 全体の総放出フラックスの内訳比較 (PSD vs TD)
    axes[0].plot(df1['TAA_Bin'], df1['Gen_Total'], '-', color='black', lw=2, label=f'総放出量 ({m1})')
    axes[0].plot(df2['TAA_Bin'], df2['Gen_Total'], '--', color='gray', lw=2, label=f'総放出量 ({m2})')
    
    axes[0].plot(df1['TAA_Bin'], df1['Gen_PSD'], '-.', color='blue', alpha=0.7, label=f'PSD ({m1})')
    axes[0].plot(df2['TAA_Bin'], df2['Gen_PSD'], ':', color='blue', alpha=0.7, label=f'PSD ({m2})')
    
    axes[0].plot(df1['TAA_Bin'], df1['Gen_TD'], '-.', color='red', alpha=0.7, label=f'TD ({m1})')
    axes[0].plot(df2['TAA_Bin'], df2['Gen_TD'], ':', color='red', alpha=0.7, label=f'TD ({m2})')
    
    axes[0].set_title("🍉 全球での放出機構別フラックスの比較", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("True Anomaly (TAA) [deg]", fontsize=11)
    axes[0].set_ylabel("放出量 [atoms/step]", fontsize=11)
    axes[0].set_yscale('log')
    axes[0].grid(True, which='both', ls='--', alpha=0.5)
    axes[0].legend(loc='lower left', fontsize=9)
    
    # 右グラフ: Dusk側上空への総放出供給フラックスの比較
    axes[1].plot(df1['TAA_Bin'], df1['Gen_Dusk_Total'], '-', color='darkorange', lw=2.5, label=f'{m1} (PSD+TD)')
    axes[1].plot(df2['TAA_Bin'], df2['Gen_Dusk_Total'], '--', color='purple', lw=2.5, label=f'{m2} (PSD+TD)')
    
    # 参考としてそれぞれのDusk側PSDとTDも薄くプロット
    axes[1].plot(df1['TAA_Bin'], df1['Gen_PSD_Dusk'], ':', color='blue', alpha=0.4, label=f'PSD Dusk ({m1})')
    axes[1].plot(df2['TAA_Bin'], df2['Gen_PSD_Dusk'], '--', color='blue', alpha=0.4, label=f'PSD Dusk ({m2})')
    axes[1].plot(df1['TAA_Bin'], df1['Gen_TD_Dusk'], ':', color='red', alpha=0.4, label=f'TD Dusk ({m1})')
    axes[1].plot(df2['TAA_Bin'], df2['Gen_TD_Dusk'], '--', color='red', alpha=0.4, label=f'TD Dusk ({m2})')
    
    axes[1].set_title("🌆 夕方側 (Dusk) に供給される総原子数の比較", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("True Anomaly (TAA) [deg]", fontsize=11)
    axes[1].set_ylabel("Dusk側への放出フラックス [atoms/step]", fontsize=11)
    axes[1].set_yscale('log')
    axes[1].grid(True, which='both', ls='--', alpha=0.5)
    axes[1].legend(loc='lower left', fontsize=9)
    
    plt.suptitle("ナトリウム表面在庫と大気密度の自己組織的バランスの定量検証", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_budget_comparison(MODELS)