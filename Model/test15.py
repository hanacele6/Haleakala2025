# -*- coding: utf-8 -*-
"""
放出位置の解析(実測のみ・再構成なし)
PSD/TD放出量 モデル間比較 (Q2.0 vs Q0.3)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
from matplotlib.ticker import MultipleLocator
import os

# ==========================================
# 設定
# ==========================================
BASE_DIR = r"./SimulationResult_202607"

MODELS = {
    "Q2.0 (Standard)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q2.0_A2.0e+07_LT190k_15yr_test2",
    "Q0.3 (Weak PSD)": "ParabolicHop_72x36_NoEq_DT100_0716_BD0.8_UG1.85_Q0.3_A2.0e+07_LT190k_15yr_test2",
}

SIDE = "Dawn"          # "Dawn" / "Dusk"

# 天頂角プロファイルで重ねるTAA
PROFILE_TAAS = [120, 140, 160, 180]


# ==========================================
# 読み込み (CSV)
# ==========================================
def load_gen(model_dir, side):
    path = os.path.join(model_dir, "band_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    df = pd.read_csv(path)

    gen_psd_col = f"Gen_PSD_{side}"
    gen_td_col = f"Gen_TD_{side}"

    bands = df[['Band_Index', 'EffCos_Lo', 'EffCos_Hi']].drop_duplicates().sort_values('Band_Index')
    band_centers = ((bands['EffCos_Lo'] + bands['EffCos_Hi']) / 2.0).values
    band_labels = [f"{lo:.2f}–{hi:.2f}" for lo, hi in zip(bands['EffCos_Lo'], bands['EffCos_Hi'])]

    def grid(col):
        return df.pivot(index='TAA_Bin', columns='Band_Index', values=col).sort_index().values

    return {
        'gen_psd': grid(gen_psd_col),
        'gen_td': grid(gen_td_col),
        'band_centers': band_centers,
        'band_labels': band_labels,
        'taa': np.sort(df['TAA_Bin'].unique()),
        'n_bands': len(band_centers),
    }


# ==========================================
# PSD vs TD モデル間重ね比較 (TAAごとのサブプロット)
# ==========================================
def plot_model_comparison_overlay(models, side, taa_list):
    # 先に全モデルのデータをロード
    model_data = {}
    for label, subdir in models.items():
        try:
            model_data[label] = load_gen(os.path.join(BASE_DIR, subdir), side)
        except Exception as e:
            print(f"データ読み込みエラー ({label}): {e}")
            continue

    if not model_data:
        print("プロットするデータがありません。")
        return

    n = len(taa_list)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 4.5 * nrow), squeeze=False)
    axes_flat = axes.flatten()

    # 線のスタイル設定 (Q2.0は実線、Q0.3は破線など)
    line_styles = {
        "Q2.0 (Standard)": {"ls": "-", "marker": "o", "alpha": 1.0},
        "Q0.3 (Weak PSD)": {"ls": "--", "marker": "^", "alpha": 0.8}
    }
    
    for pi, target in enumerate(taa_list):
        ax = axes_flat[pi]
        
        for label, d in model_data.items():
            taa_axis = d['taa']
            bc = d['band_centers']
            
            ti = np.argmin(np.abs(taa_axis - target))
            actual = taa_axis[ti]
            
            psd = d['gen_psd'][ti, :]
            td = d['gen_td'][ti, :]
            
            style = line_styles.get(label, {"ls": "-", "marker": "x", "alpha": 1.0})
            
            # PSDのプロット (青系)
            ax.plot(bc, psd, ls=style["ls"], marker=style["marker"], 
                    color='royalblue', ms=5, lw=2, alpha=style["alpha"], 
                    label=f'PSD [{label}]')
            
            # TDのプロット (赤系)
            ax.plot(bc, td, ls=style["ls"], marker=style["marker"], 
                    color='crimson', ms=5, lw=2, alpha=style["alpha"], 
                    label=f'TD [{label}]')

        # 線形スケール＋指数表記
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)

        ax.set_title(f'TAA = {actual}°', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, which='both', ls='--', alpha=0.4)
        
        # 凡例は最初のサブプロットのみに表示してスッキリさせる
        if pi == 0:
            ax.legend(loc='upper left', fontsize=9, ncol=1)

    # 余ったサブプロットを非表示
    for pi in range(n, len(axes_flat)):
        axes_flat[pi].set_visible(False)

    # 軸ラベルの設定
    for pi in range(n):
        r, cc = divmod(pi, ncol)
        if r == nrow - 1 or pi + ncol >= n:
            axes_flat[pi].set_xlabel('eff_cos [0=ターミネーター → 1=SSP]', fontsize=11)
        if cc == 0:
            axes_flat[pi].set_ylabel('放出量 [atoms]', fontsize=11)

    fig.suptitle(f'PSD vs TD 放出量 モデル間比較 ({side}側) [線形スケール]', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 指定したTAAごとのサブプロットで、複数モデルのPSD/TDを比較
    plot_model_comparison_overlay(MODELS, SIDE, PROFILE_TAAS)