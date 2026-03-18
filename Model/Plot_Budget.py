# -*- coding: utf-8 -*-
"""
==============================================================================
水星ナトリウム外気圏シミュレーション
Budget Analysis 可視化スクリプト

概要:
    simulation_with_budget.py で出力された 'budget_statistics_per_taa.csv' を読み込み、
    TAAごとの生成(Source)と消滅(Loss)の内訳を積み上げグラフで表示・保存する。

作成日: 2026/01/30
==============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ==========================================================
# 設定: ここにCSVファイルのパスを指定してください
# ==========================================================
# 例: r"./SimulationResult_202512/ParabolicHop_.../budget_statistics_per_taa.csv"
#CSV_FILE_PATH = r"./SimulationResult_202512/ParabolicHop_72x36_NoEq_DT100_0128_0.4Denabled_2.7_LowestQ_test/budget_statistics_per_taa.csv"
#CSV_FILE_PATH = r"./SimulationResult_202602/ParabolicHop_72x36_NoEq_DT100_0213_0.4Denabled_2.7_LowestQ_Bounce575K/budget_statistics_per_taa.csv"
#CSV_FILE_PATH = r"./SimulationResult_202602/HeteroSurf_72x36_NoEq_DT100_UDist_Refill0.1/budget_statistics_per_taa.csv"
CSV_FILE_PATH = r"./SimulationResult_202603/ParabolicHop_72x36_NoEq_DT100_0317_Multi_0.4Denabled_1.85&2.7_OnlyLowestQ_Bouncetau30s_A2.0_LongLT/budget_statistics_per_taa.csv"

# 図の保存設定
SAVE_FIGS = True
OUTPUT_PREFIX = "BudgetPlot_"

# ==========================================================
# ★追加設定: 縦線を引きたいTAAを指定
# 形式: {TAA(deg): "ラベルテキスト"}
# ラベルが不要な場合は空文字 "" にしてください
# ==========================================================
TARGET_TAA_LINES = {
    150: "",
}

# ==========================================================
# プロット用スタイル設定
# ==========================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 色定義
COLORS = {
    'PSD': '#E15759',  # Red
    'TD': '#4E79A7',  # Blue
    'SWS': '#59A14F',  # Green
    'MMV': '#79706E',  # Gray

    'Stuck': '#EDc948',  # Yellow/Orange
    'Ionized': '#B07AA1',  # Purple
    'Escaped': '#FF9DA7',  # Pink
}


def draw_vertical_lines(ax, lines_dict, show_labels=False):
    """
    指定されたTAAに縦線を描画するヘルパー関数

    Args:
        ax: matplotlibのaxesオブジェクト
        lines_dict: {taa: label} の辞書
        show_labels: テキストラベルを表示するかどうか (上段グラフ向け)
    """
    if not lines_dict:
        return

    trans = ax.get_xaxis_transform()  # xはデータ座標、yは軸座標(0-1)を使用

    for taa, label in lines_dict.items():
        # 縦線を描画
        ax.axvline(x=taa, color='black', linestyle='--', linewidth=1.2, alpha=0.7, zorder=10)

        # ラベルを描画 (show_labelsがTrue かつ ラベル文字がある場合)
        if show_labels and label:
            # グラフの上端少し上に表示
            ax.text(taa, 1.02, label, transform=trans,
                    ha='center', va='bottom', fontsize=10,
                    color='black', rotation=45, fontweight='bold')


def plot_source_budget(df):
    """生成プロセス(Source)の積み上げグラフを描画"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    taa = df['TAA_Bin']

    # データ準備
    sources = ['PSD', 'TD', 'SWS', 'MMV']
    labels = ['PSD (Photon-Stimulated)', 'TD (Thermal Desorption)', 'SWS (Solar Wind)', 'MMV (Micrometeoroid)']
    data_abs = [df[f'Gen_{s}'] for s in sources]
    data_pct = [df[f'Pct_{s}'] for s in sources]
    colors = [COLORS[s] for s in sources]

    # --- 上段: 絶対量 (Stackplot) ---
    ax1.stackplot(taa, data_abs, labels=labels, colors=colors, alpha=0.85)

    ax1.set_ylabel('Generation Rate (atoms)', fontsize=14)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ★縦線追加 (上段にはラベルも表示)
    draw_vertical_lines(ax1, TARGET_TAA_LINES, show_labels=True)

    # Y軸を科学的表記に
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # --- 下段: 割合 (Stackplot) ---
    ax2.stackplot(taa, data_pct, labels=labels, colors=colors, alpha=0.85)

    ax2.set_ylabel('Contribution (%)', fontsize=14)
    ax2.set_xlabel('True Anomaly Angle (TAA) [deg]', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ★縦線追加 (下段は線のみ)
    draw_vertical_lines(ax2, TARGET_TAA_LINES, show_labels=False)

    # TAA軸の装飾
    setup_taa_axis(ax2)

    return fig


def plot_loss_budget(df):
    """消滅プロセス(Loss)の積み上げグラフを描画"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    taa = df['TAA_Bin']

    # データ準備
    losses = ['Stuck', 'Ionized', 'Escaped']
    labels = ['Surface Stuck', 'Photo-Ionization', 'Jeans Escape / Loss']
    data_abs = [df[f'Loss_{l}'] for l in losses]
    data_pct = [df[f'Pct_{l}'] for l in losses]
    colors = [COLORS[l] for l in losses]

    # --- 上段: 絶対量 (Stackplot) ---
    ax1.stackplot(taa, data_abs, labels=labels, colors=colors, alpha=0.85)

    ax1.set_ylabel('Loss Rate (Weight Sum)', fontsize=14)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ★縦線追加 (上段にはラベルも表示)
    draw_vertical_lines(ax1, TARGET_TAA_LINES, show_labels=True)

    # Y軸を科学的表記に
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # --- 下段: 割合 (Stackplot) ---
    ax2.stackplot(taa, data_pct, labels=labels, colors=colors, alpha=0.85)

    ax2.set_ylabel('Fraction (%)', fontsize=14)
    ax2.set_xlabel('True Anomaly Angle (TAA) [deg]', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # ★縦線追加 (下段は線のみ)
    draw_vertical_lines(ax2, TARGET_TAA_LINES, show_labels=False)

    # TAA軸の装飾
    setup_taa_axis(ax2)

    return fig


def setup_taa_axis(ax):
    """TAA軸の目盛りとラベルを整形"""
    ax.set_xlim(0, 360)
    xticks = [0, 90, 180, 270, 360]
    ax.set_xticks(xticks)


def main():
    # CSV読み込み
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Error: CSV file not found at {CSV_FILE_PATH}")
        print("Please check the path or run the simulation first.")
        # テスト用にダミーデータを作成するか、終了する
        return

    print(f"Loading data from {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # --- 図1: Source Budget ---
    print("Plotting Source Budget...")
    fig_source = plot_source_budget(df)
    if SAVE_FIGS:
        fname = OUTPUT_PREFIX + "Source.png"
        fig_source.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    plt.show()

    # --- 図2: Loss Budget ---
    print("Plotting Loss Budget...")
    fig_loss = plot_loss_budget(df)
    if SAVE_FIGS:
        fname = OUTPUT_PREFIX + "Loss.png"
        fig_loss.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
    plt.show()


if __name__ == "__main__":
    main()