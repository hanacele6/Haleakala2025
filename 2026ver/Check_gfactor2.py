import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_g_factor_variation(dawn_excel_path, dusk_excel_path):
    print("--- g-factorの変動プロットを開始します ---")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    dfs = []

    # データの読み込み
    for path, label, color in zip([dawn_excel_path, dusk_excel_path], 
                                  ['Dawn', 'Dusk'], 
                                  ['blue', 'red']):
        if Path(path).exists():
            df = pd.read_excel(path)
            taa_col = df.columns[2]  # 3列目がTAA
            df_plot = df.dropna(subset=[taa_col, 'g_factor_Calculated'])
            
            # g-factorをプロット
            ax.plot(df_plot[taa_col], df_plot['g_factor_Calculated'], 
                    marker='.', linestyle='', color=color, alpha=0.5, label=label)
            
            dfs.append(df_plot[[taa_col, 'g_factor_Calculated']].rename(columns={taa_col: 'TAA'}))

    # なめらかなトレンド線を描画
    if dfs:
        df_all = pd.concat(dfs).dropna()
        df_all['TAA_int'] = df_all['TAA'].round(0)
        df_smooth = df_all.groupby('TAA_int')['g_factor_Calculated'].mean().reset_index()
        
        ax.plot(df_smooth['TAA_int'], df_smooth['g_factor_Calculated'], 
                color='black', linestyle='-', linewidth=2, label='g-factor Trend')

    # グラフの装飾
    ax.set_title("Variation of g-factor vs True Anomaly Angle", fontsize=14)
    ax.set_xlabel("True Anomaly Angle (°)", fontsize=12)
    ax.set_ylabel("g-factor (photons/s/atom)", fontsize=12)
    
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 60))
    
    # 論文で指摘されている「g-factorが最大になるポイント(赤の縦破線)」の目安
    ax.axvline(70, color='red', linestyle=':', alpha=0.8, label='Max Solar Pressure (~70°)')
    ax.axvline(290, color='red', linestyle=':', alpha=0.8, label='Max Solar Pressure (~290°)')

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper center', fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ★ Excelのパスを指定
    dawn_file = "C:/Users/hanac/univ/Mercury/Dawn_Brightness.xlsx"
    dusk_file = "C:/Users/hanac/univ/Mercury/Dusk_Brightness.xlsx"
    
    plot_g_factor_variation(dawn_file, dusk_file)