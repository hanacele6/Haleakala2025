# -*- coding: utf-8 -*-
"""
Budget Analysis 可視化スクリプト (Rate変換版)
Weight Sum [atoms] を 滞在時間 [s] で割って Source Rate [atoms/s] に変換して表示する
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# === 設定 ===
CSV_FILE_PATH = "budget_statistics_per_taa.csv"  # 出力されたCSVのパス
SAVE_FIGS = True
OUTPUT_PREFIX = "BudgetRate_"

# === 物理定数 (軌道計算用) ===
AU = 1.496e11
MERCURY_SEMI_MAJOR_AXIS = 0.387098 * AU
MERCURY_ECCENTRICITY = 0.205630
GM_MERCURY = 2.2032e13
# 日心重力定数 (GM_Sun) ※水星の公転速度計算に必要
GM_SUN = 1.3271244e20


def calculate_residence_time_per_degree():
    """
    TAA 0~359度について、各1度分の通過にかかる時間 dt [s] を計算する
    (ケプラーの第2法則: r^2 * d(theta)/dt = h を利用)
    """
    a = MERCURY_SEMI_MAJOR_AXIS
    e = MERCURY_ECCENTRICITY

    # 角運動量 (特定角運動量 h = sqrt(G*M*a*(1-e^2)))
    # ※ここは公転なので中心天体は太陽(GM_SUN)
    h = np.sqrt(GM_SUN * a * (1 - e ** 2))

    dt_list = []
    d_theta = np.deg2rad(1.0)  # 1度分のラジアン

    for deg in range(360):
        # 区間中心の角度で近似
        taa_rad = np.deg2rad(deg + 0.5)

        # 距離 r
        r = a * (1 - e ** 2) / (1 + e * np.cos(taa_rad))

        # dt = (r^2 / h) * d_theta
        dt = (r ** 2 / h) * d_theta
        dt_list.append(dt)

    return np.array(dt_list)


# プロット設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
COLORS = {
    'PSD': '#E15759', 'TD': '#4E79A7', 'SWS': '#59A14F', 'MMV': '#79706E',
    'Stuck': '#EDc948', 'Ionized': '#B07AA1', 'Escaped': '#FF9DA7',
}


def plot_budget_rate(df, dt_array, mode='Source'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    taa = df['TAA_Bin']

    if mode == 'Source':
        keys = ['PSD', 'TD', 'SWS', 'MMV']
        col_prefix = 'Gen_'
        title = 'Source Rate vs TAA'
        ylabel = 'Source Rate (atoms/s)'
    else:
        keys = ['Stuck', 'Ionized', 'Escaped']
        col_prefix = 'Loss_'
        title = 'Loss Rate vs TAA'
        ylabel = 'Loss Rate (atoms/s)'

    # --- [変換] atoms -> atoms/s ---
    # 各ビンの総量(atoms)を、滞在時間(dt)で割る
    data_rate = []
    for k in keys:
        # CSVの生データ(atoms) / 時間(s)
        rate_vals = df[f'{col_prefix}{k}'] / dt_array
        data_rate.append(rate_vals)

    # 割合の計算 (Rateベースで再計算しても、atomsベースと同じだが念のため)
    total_rate = np.sum(data_rate, axis=0)
    # ゼロ除算回避
    safe_total = np.where(total_rate > 0, total_rate, 1.0)
    data_pct = [(d / safe_total * 100.0) for d in data_rate]

    colors = [COLORS[k] for k in keys]

    # 上段: Rate (Stackplot)
    ax1.stackplot(taa, data_rate, labels=keys, colors=colors, alpha=0.85)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # 下段: 割合 (%)
    ax2.stackplot(taa, data_pct, labels=keys, colors=colors, alpha=0.85)
    ax2.set_ylabel('Fraction (%)', fontsize=14)
    ax2.set_xlabel('True Anomaly Angle (TAA) [deg]', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 360)
    ax2.set_xticks([0, 90, 180, 270, 360])
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ラベル
    #trans = ax2.get_xaxis_transform()
    #for x, txt in [(0, 'Peri'), (180, 'Aphelion'), (360, 'Peri')]:
    #    ax2.text(x, -0.15, txt, transform=trans, ha='center', color='blue')

    return fig


def main():
    if not os.path.exists(CSV_FILE_PATH):
        print("CSV not found.")
        return

    df = pd.read_csv(CSV_FILE_PATH)

    # 滞在時間を計算
    dt_array = calculate_residence_time_per_degree()

    # プロット
    fig1 = plot_budget_rate(df, dt_array, 'Source')
    if SAVE_FIGS: fig1.savefig(OUTPUT_PREFIX + "Source.png", dpi=150, bbox_inches='tight')

    fig2 = plot_budget_rate(df, dt_array, 'Loss')
    if SAVE_FIGS: fig2.savefig(OUTPUT_PREFIX + "Loss.png", dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()