# -*- coding: utf-8 -*-
"""
TD放出の「在庫 × 率」分解 と 遠日点対称性テスト (在庫バリエーション版)

ユーザーの要求により、他のグラフは生成せず、
「在庫 (Stock)」のTAA依存性を以下の3パターンで可視化する。
  1. 昼面全体 (Dawn + Dusk)
  2. DAWN側のみ
  3. DUSK側のみ

物理的理由:
  渋滞在庫は一方通行で Dawn半球を通過する塊であり、
  TAA=180 付近で SSP に到達して Dusk 側へ抜けていく。
  よって TAA=160(到達中) と TAA=200(通過後) は状態が全く違う。
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
COLORS = {"Q2.0 (Standard)": "crimson", "Q0.3 (Weak PSD)": "steelblue"}

# [B] 対称性テストで見る遠日点からのずれ幅
MIRROR_OFFSETS = [10, 20, 30, 40, 50, 60, 80]

# --- 物理定数(本体 mkNaColumnDensity と厳密一致) ---
KB = 1.380649e-23
EV = 1.602e-19
TEMP_BASE = 100.0
TEMP_AMP = 600.0
A_AU = 0.387098
ECC = 0.205630
TD_PREFACTOR = 1e13
U_EV = 1.85


# ==========================================
# 物理式
# ==========================================
def au_of_taa(taa_deg):
    return A_AU * (1 - ECC**2) / (1 + ECC * np.cos(np.deg2rad(taa_deg)))


def temperature(eff_cos, au):
    ec = np.clip(eff_cos, 0, None)
    return TEMP_BASE + TEMP_AMP * (ec ** 0.25) * np.sqrt(0.306 / au)


def rate_td(eff_cos, au, u_ev=U_EV):
    T = temperature(eff_cos, au)
    expo = -(u_ev * EV) / (KB * T)
    return np.where(expo >= -700, TD_PREFACTOR * np.exp(np.clip(expo, -700, None)), 0.0)


# ==========================================
# 読み込み
# ==========================================
def load_band(model_dir, side):
    path = os.path.join(model_dir, "band_statistics_per_taa.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"見つかりません: {path}")
    df = pd.read_csv(path)

    stock_col = f"Stock_{side}" if f"Stock_{side}" in df.columns else f"Sweep_{side}"
    if 'Step_Count' in df.columns:
        sc = df['Step_Count'].replace(0, np.nan)
        df['Stock'] = df[stock_col] / sc
    else:
        df['Stock'] = df[stock_col]
    df['Stock'] = df['Stock'].fillna(0.0)
    df['Gen_TD'] = df[f'Gen_TD_{side}']
    df['Gen_PSD'] = df[f'Gen_PSD_{side}']

    bands = df[['Band_Index', 'EffCos_Lo', 'EffCos_Hi']].drop_duplicates().sort_values('Band_Index')
    bc = ((bands['EffCos_Lo'] + bands['EffCos_Hi']) / 2.0).values
    labels = [f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(bands['EffCos_Lo'], bands['EffCos_Hi'])]

    def grid(col):
        return df.pivot(index='TAA_Bin', columns='Band_Index', values=col).sort_index().values

    return {
        'stock': grid('Stock'),
        'gen_td': grid('Gen_TD'),
        'gen_psd': grid('Gen_PSD'),
        'bc': bc,
        'labels': labels,
        'taa': np.sort(df['TAA_Bin'].unique()),
    }


def row_at(d, taa_target):
    ti = int(np.argmin(np.abs(d['taa'] - taa_target)))
    return ti, d['taa'][ti]


# ==========================================
# [C] 可視化: 在庫のTAA依存 (バリエーション追加)
# ==========================================
def plot_stock_variations(models):
    # 3つの独立したFigureとAxesを作成 (DUSK側用を追加)
    fig1, ax1 = plt.subplots(figsize=(10, 4.5))
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))

    for label, subdir in models.items():
        try:
            # "Dawn" と "Dusk" の両方のデータを読み込む
            d_dawn = load_band(os.path.join(BASE_DIR, subdir), "Dawn")
            d_dusk = load_band(os.path.join(BASE_DIR, subdir), "Dusk")
        except Exception as e:
            print(f"[エラー] {label}: {e}")
            continue
        
        # TAAと最上位バンドの在庫を抽出
        taa = d_dawn['taa']
        stock_top_dawn = d_dawn['stock'][:, -1]
        stock_top_dusk = d_dusk['stock'][:, -1]
        c = COLORS.get(label)

        # それぞれのAxesにプロット

        # グラフ 1: 昼面全体 (Dawn + Dusk)
        stock_top_full = stock_top_dawn + stock_top_dusk
        ax1.plot(taa, stock_top_full, '-', color=c, lw=2, label=f'{label}')

        # グラフ 2: DAWN側のみ
        ax2.plot(taa, stock_top_dawn, '-', color=c, lw=2, label=f'{label}')

        # グラフ 3: DUSK側のみ (追加)
        ax3.plot(taa, stock_top_dusk, '-', color=c, lw=2, label=f'{label}')

    # 各グラフの設定
    titles = [
        f'在庫 Stock — 昼面全体 (Dawn + Dusk)',
        f'在庫 Stock — DAWN側のみ',
        f'在庫 Stock — DUSK側のみ',
    ]
    
    figures = [fig1, fig2, fig3]
    axes = [ax1, ax2, ax3]

    for fig, ax, t in zip(figures, axes, titles):
        ax.axvline(180, color='gray', ls=':', alpha=0.7, label='遠日点(180)')
        ax.axvline(165, color='green', ls='--', alpha=0.5, label='Dawn側ピーク目安(165)')
        #ax.axvline(195, color='orange', ls='--', alpha=0.5, label='Dusk側通過目安(195)') # Duskの目安線も追加
        
        ax.set_title(t, fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=9)
        
        ax.set_xlabel('TAA [deg]')
        ax.set_ylabel('Stock')
        ax.set_xlim(0, 360)
        ax.xaxis.set_major_locator(MultipleLocator(60))
        
        fig.tight_layout()

    # plt.show() を1回呼ぶことで、作成した3つのウィンドウが同時に開きます
    plt.show()

if __name__ == "__main__":
    # 他のテストは実行しない
    # decomposition_test(MODELS, SIDE, TAA_A, TAA_B)
    # symmetry_test(MODELS, SIDE, MIRROR_OFFSETS)
    
    # 在庫バリエーションのプロットのみ実行
    plot_stock_variations(MODELS)