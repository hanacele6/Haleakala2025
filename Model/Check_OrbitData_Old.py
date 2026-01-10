#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_orbit_cyclic_v6.py
==============================================================================
水星の二体問題データ (orbit2025_v6.txt) は「1公転分」のみです。
これを複数年シミュレーションする場合、
「1公転ごとに自転が1.5回転するため、太陽直下点は180度ずれる」
という性質を考慮する必要があります。

このスクリプトは、1年分のデータを元に「3公転分」のアニメーションを生成し、
以下の挙動を確認します。
1. 近日点での逆行（秤動）
2. 1年経過するごとに、太陽直下点(水色)と経度0度(赤線)の関係が反転すること
   (0年目: 0度が正面 -> 1年目: 180度が正面 -> 2年目: 0度が正面)
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# --- 設定 ---
FILENAME = 'orbit2025_v6.txt'
MERCURY_RADIUS_AU = 0.05
AU_TO_KM = 1.496e8
MERCURY_SPIN_PERIOD_DAY = 58.646
ORBITAL_PERIOD_DAY = 87.969

# 3公転分シミュレーションする
SIMULATION_YEARS = 3.2


def load_v6_data_cyclic(filename):
    if not os.path.exists(filename):
        print(f"[Error] '{filename}' が見つかりません。", file=sys.stderr)
        return None

    try:
        # data cols: 0:TAA, 1:AU, 2:Time, 3:Vr, 4:Vt, 5:SSL
        data = np.loadtxt(filename, comments='#')

        # SSLのunwrap (補間計算のため)
        ssl_deg = data[:, 5]
        data[:, 5] = np.degrees(np.unwrap(np.radians(ssl_deg)))

        return data
    except Exception as e:
        print(f"[Error] ロード失敗: {e}", file=sys.stderr)
        return None


def get_cyclic_state(t_total, data, t_max):
    """
    累積時間 t_total における状態を、1年分のデータ data から計算する。
    周回数に応じて SSL を 180度シフトさせる。
    """
    # 現在が何周目か (0, 1, 2...)
    cycle = int(t_total / t_max)

    # 1年以内の時間に正規化
    t_in_cycle = t_total % t_max

    # 線形補間でデータを取得
    # data[:, 2] が Time[s]
    times = data[:, 2]

    # 軌道形状(r, taa)や速度(vt)は周期的 -> そのまま補間
    r_au = np.interp(t_in_cycle, times, data[:, 1])
    taa = np.interp(t_in_cycle, times, data[:, 0])
    vt_ms = np.interp(t_in_cycle, times, data[:, 4])

    # 太陽直下点経度(SSL) の計算
    # 1周分のデータにおけるSSL
    ssl_base = np.interp(t_in_cycle, times, data[:, 5])

    # ★重要★
    # 1周(87.969日)で自転(58.646日)は 1.5回転する。
    # つまり、SSL (TAA - Rotation) は 1周ごとに -0.5回転 (-180度) ずれる。
    # なので、周回数(cycle) * 180度 を引くことで正しいSSLになる。
    ssl_cyclic = ssl_base - (cycle * 180.0)

    return taa, r_au, vt_ms, ssl_cyclic, cycle


def animate_cyclic_check(data):
    # データの最大時間 (1公転周期)
    t_max = data[-1, 2]  # 秒

    # アニメーション用の時間ステップ生成 (3.2年分)
    total_duration = t_max * SIMULATION_YEARS
    steps = 400  # フレーム数
    time_points = np.linspace(0, total_duration, steps)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    ax.set_facecolor('black')
    ax.set_title(f"Mercury 3-Year Cycle Check\nSwitching 0deg <-> 180deg every year")
    ax.set_aspect('equal')

    # スケール設定
    max_r = np.max(data[:, 1]) * 1.1
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)

    # 静的要素
    ax.plot(0, 0, 'o', color='yellow', markersize=20, label='Sun')

    # 軌道線（目安）
    theta = np.radians(data[:, 0])
    r = data[:, 1]
    ax.plot(r * np.cos(theta), r * np.sin(theta), ':', color='gray', alpha=0.5)

    # 動的要素
    mercury, = ax.plot([], [], 'o', color='lightgray', markersize=25, zorder=5)
    prime_meridian, = ax.plot([], [], '-', color='red', linewidth=2, zorder=6, label='Prime Meridian (0 deg)')
    ssl_marker, = ax.plot([], [], 'o', color='cyan', markersize=6, zorder=7, label='Sub-Solar Point')

    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, color='white',
                        verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6))

    ax.legend(loc='lower right')

    def update(frame):
        t_current = time_points[frame]

        # 補間計算
        taa, r_au, vt, ssl, cycle = get_cyclic_state(t_current, data, t_max)

        # 座標計算
        rad = np.radians(taa)
        x = r_au * np.cos(rad)
        y = r_au * np.sin(rad)

        # 描画更新
        mercury.set_data([x], [y])

        # プライムメリディアン (赤線: 経度0度)
        # 幾何学的関係: TAA + 180 - SSL = PMの角度 (慣性系)
        pm_angle_deg = (taa + 180.0 - ssl)
        pm_rad = np.radians(pm_angle_deg)

        r_vis = MERCURY_RADIUS_AU * 0.8
        pm_x = x + r_vis * np.cos(pm_rad)
        pm_y = y + r_vis * np.sin(pm_rad)
        prime_meridian.set_data([x, pm_x], [y, pm_y])

        # 太陽直下点 (水色点)
        ssl_mark_rad = pm_rad + np.radians(ssl)
        ssl_x = x + r_vis * np.cos(ssl_mark_rad)
        ssl_y = y + r_vis * np.sin(ssl_mark_rad)
        ssl_marker.set_data([ssl_x], [ssl_y])

        # 情報表示
        days = t_current / (24 * 3600)
        ssl_norm = ssl % 360
        if ssl_norm > 180: ssl_norm -= 360

        info_text.set_text(f'Time: {days:.1f} days\n'
                           f'Year (Cycle): {cycle}\n'
                           f'SSL: {ssl_norm:.1f} deg\n'
                           f'(Every year SSL shifts 180 deg)')

        return mercury, prime_meridian, ssl_marker, info_text

    ani = FuncAnimation(fig, update, frames=steps, interval=50, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    v6_data = load_v6_data_cyclic(FILENAME)
    if v6_data is not None:
        print("Start 3-year cyclic animation...")
        animate_cyclic_check(v6_data)