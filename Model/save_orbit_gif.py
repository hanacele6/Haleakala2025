# -*- coding: utf-8 -*-
"""
save_orbit_simple_black.py
==============================================================================
・背景を黒に統一
・赤い線（自転基準線）を削除
・太陽直下点（水色）とTAA情報のみを表示
・skip=100, FPS=10 (ゆっくり)
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# --- 設定パラメータ ---
SKIP_STEP = 100  # 100行ごとに描画
OUTPUT_FPS = 10  # 再生速度 (10fps = ゆっくり)
OUTPUT_GIF = "mercury_orbit_simple.gif"

# 定数
MERCURY_RADIUS_AU = 0.05
AU_TO_KM = 1.496e8


def load_orbit_data(filename='orbit2025_spice_unwrapped.txt'):
    if not os.path.exists(filename):
        print(f"[Error] '{filename}' が見つかりません。", file=sys.stderr)
        return None
    try:
        return np.loadtxt(filename, skiprows=1)
    except Exception as e:
        print(f"[Error] ロード失敗: {e}", file=sys.stderr)
        return None


def cut_one_orbit(data):
    """開始から360度（1周分）でカット"""
    taa = data[:, 0]
    diff = np.diff(taa)
    diff = np.where(diff < -300, diff + 360, diff)
    cumulative = np.cumsum(np.insert(diff, 0, 0))

    idx_over = np.where(cumulative >= 360.0)[0]
    if len(idx_over) > 0:
        return data[:idx_over[0] + 1]
    return data


def save_orbit_animation(data):
    # 1. データを1周分にカット
    data = cut_one_orbit(data)

    # 2. skip=100 で間引く
    data_sub = data[::SKIP_STEP]
    print(f"フレーム数: {len(data_sub)} (skip={SKIP_STEP})")

    # --- 配列展開 ---
    taa_deg = data_sub[:, 0]
    r_au = data_sub[:, 1]
    # 赤い線が不要になったので ssl_deg (自転角度) のデータは使いませんが、
    # 太陽直下点の計算（幾何学的に太陽方向）のために角度計算は行います。

    # 座標計算
    taa_rad = np.radians(taa_deg)
    X_merc = r_au * np.cos(taa_rad)
    Y_merc = r_au * np.sin(taa_rad)

    # --- プロット設定 ---
    # figureの背景も黒にする
    fig = plt.figure(figsize=(8, 8), facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('black')

    # 軸や枠線を完全に消す
    ax.axis('off')

    # 範囲設定
    limit = np.max(r_au) * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # --- 静的要素 ---
    # 1. 太陽
    ax.plot(0, 0, 'o', color='orange', markersize=60, zorder=10)
    ax.text(0, 0, "SUN", color='black', fontsize=12, fontweight='bold',
            ha='center', va='center', zorder=11)

    # 2. 軌道ライン (薄いグレー)
    ax.plot(X_merc, Y_merc, ':', color='#444444', alpha=0.6, linewidth=1.5)

    # --- 動的要素 ---
    # 水星本体
    body, = ax.plot([], [], 'o', color='lightgray', markersize=40, zorder=5)

    # 太陽直下点 (水色点) - 赤い線がなくてもこれだけは残す
    dot, = ax.plot([], [], 'o', color='cyan', markersize=10, zorder=7)

    # TAA表示テキスト (左上)
    text_taa = ax.text(0.05, 0.9, '', transform=ax.transAxes, color='white',
                       fontsize=24, fontweight='bold')

    # 更新関数
    def update(i):
        if i % 10 == 0:
            print(f"\rGenerating: {i}/{len(X_merc)}", end="")

        x, y = X_merc[i], Y_merc[i]
        curr_taa = taa_deg[i]

        # 水星位置
        body.set_data([x], [y])

        # 太陽直下点 (常に太陽の方角、つまり (0,0) の方向を向く)
        # 水星から見て太陽の方向 = 現在の角度 + 180度
        angle_to_sun = taa_rad[i] + np.pi

        vis_r = MERCURY_RADIUS_AU * 0.8
        dx = vis_r * np.cos(angle_to_sun)
        dy = vis_r * np.sin(angle_to_sun)

        dot.set_data([x + dx], [y + dy])

        # テキスト更新 (TAAのみ)
        text_taa.set_text(f"TAA: {curr_taa:.1f}°")

        return body, dot, text_taa

    # --- 保存実行 ---
    print("アニメーション生成開始...")
    ani = FuncAnimation(fig, update, frames=len(X_merc), interval=100, blit=True)

    ani.save(OUTPUT_GIF, writer='pillow', fps=OUTPUT_FPS, dpi=100)
    print(f"\n完了！ '{OUTPUT_GIF}' を保存しました。")


if __name__ == '__main__':
    d = load_orbit_data()
    if d is not None:
        save_orbit_animation(d)