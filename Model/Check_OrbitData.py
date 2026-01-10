# -*- coding: utf-8 -*-
"""
check_orbit_plot_v4.py
==============================================================================
generate_orbit_spice.pyで生成された水星軌道データ
(orbit2025_spice.txt)を視覚的にチェックするスクリプト。

【今回の修正】
- 太陽直下点 (Sub-Solar Point) のマーカーを復活・強調表示。
- 太陽から見た自転速度（見かけの回転）のグラフと連動。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# 定数
MERCURY_RADIUS_AU = 0.05  # 描画上の半径 (AUスケール)
AU_TO_KM = 1.496e8  # 1 AU = 1.496e8 km
MERCURY_SPIN_PERIOD_DAY = 58.646  # 水星の自転周期 [day]


def load_orbit_data(filename='orbit2025_spice_unwrapped.txt'):
    if not os.path.exists(filename):
        print(f"[Error] '{filename}' が見つかりません。", file=sys.stderr)
        return None
    try:
        data = np.loadtxt(filename, skiprows=1)
        return data
    except Exception as e:
        print(f"[Error] ロード失敗: {e}", file=sys.stderr)
        return None


def animate_orbit_check(data):
    # --- データ展開 ---
    taa_deg = data[:, 0]  # TAA [deg]
    r_au = data[:, 1]  # 距離 [AU]
    vt_ms = data[:, 4]  # 接線速度 Vt [m/s]
    ssl_deg = data[:, 5]  # SubSolarLon [deg]

    # --- 物理量の計算: 見かけの回転速度 ---
    # 1. 公転角速度
    r_km = r_au * AU_TO_KM
    vt_km_s = vt_ms / 1000.0
    omega_orb_rad_s = vt_km_s / r_km
    omega_orb_deg_day = np.degrees(omega_orb_rad_s) * (24 * 3600)

    # 2. 自転角速度
    spin_rate_deg_day = 360.0 / MERCURY_SPIN_PERIOD_DAY

    # 3. 太陽から見た見かけの回転速度 (正=順行, 負=逆行)
    apparent_rotation_speed = spin_rate_deg_day - omega_orb_deg_day

    # --- 描画準備 ---
    taa_rad = np.radians(taa_deg)
    X_merc = r_au * np.cos(taa_rad)
    Y_merc = r_au * np.sin(taa_rad)

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax_orbit = fig.add_subplot(gs[0])
    ax_graph = fig.add_subplot(gs[1])

    # --- [上段] 軌道プロット ---
    ax_orbit.set_facecolor('black')
    ax_orbit.set_title("Mercury Orbit Check (Red Line: 0deg Lon, Cyan Dot: Sub-Solar Point)")
    ax_orbit.set_aspect('equal', adjustable='box')
    max_r = np.max(r_au) * 1.1
    ax_orbit.set_xlim(-max_r, max_r)
    ax_orbit.set_ylim(-max_r, max_r)

    # 静的要素
    ax_orbit.plot(0, 0, 'o', color='yellow', markersize=20, label='Sun')
    ax_orbit.plot(X_merc, Y_merc, ':', color='gray', alpha=0.4, linewidth=0.5)

    # 動的要素
    mercury_body, = ax_orbit.plot([], [], 'o', color='lightgray', markersize=25, zorder=5)

    # 赤線: プライムメリディアン (経度0度) - 自転とともに回る
    prime_meridian, = ax_orbit.plot([], [], '-', color='red', linewidth=2, zorder=6, label='Prime Meridian (0 deg)')

    # 水色点: 太陽直下点 (Sub-Solar Point) - 常に太陽方向を向くはず
    ssl_marker, = ax_orbit.plot([], [], 'o', color='cyan', markersize=6, zorder=7, label='Sub-Solar Point')

    # 情報テキスト
    info_text = ax_orbit.text(0.02, 0.98, '', transform=ax_orbit.transAxes, color='white',
                              verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6))

    ax_orbit.legend(loc='upper right', fontsize='small')

    # --- [下段] グラフ ---
    ax_graph.set_title("Apparent Rotation Speed (Spin - Orbit)")
    ax_graph.set_xlabel("Step")
    ax_graph.set_ylabel("Speed [deg/day]")
    ax_graph.grid(True, linestyle='--', alpha=0.5)

    steps = np.arange(len(data))
    ax_graph.axhline(0, color='white', linewidth=1)
    ax_graph.plot(steps, apparent_rotation_speed, color='cyan')
    ax_graph.fill_between(steps, apparent_rotation_speed, 0,
                          where=(apparent_rotation_speed < 0),
                          color='red', alpha=0.3, label='Retrograde (Reverse)')

    current_time_bar = ax_graph.axvline(x=0, color='yellow', linestyle='-', linewidth=2)
    ax_graph.legend()

    # --- アニメーション更新 ---
    def update(frame):
        X_f, Y_f = X_merc[frame], Y_merc[frame]
        taa_f = taa_deg[frame]
        ssl_f = ssl_deg[frame]
        curr_app_w = apparent_rotation_speed[frame]

        # 色設定: 逆行中は水星本体を赤く
        if curr_app_w < 0:
            merc_color = 'salmon'
            status_str = "RETROGRADE (Reverse)"
        else:
            merc_color = 'lightgray'
            status_str = "PROGRADE (Normal)"

        # 1. 水星本体
        mercury_body.set_data([X_f], [Y_f])
        mercury_body.set_color(merc_color)

        # 2. プライムメリディアン (赤線: 経度0度)
        # 慣性系角度 = TAA + 180 - SSL
        pm_angle_deg = (taa_f + 180.0 - ssl_f) % 360.0
        pm_rad = np.radians(pm_angle_deg)

        r_vis = MERCURY_RADIUS_AU * 0.8
        pm_x = X_f + r_vis * np.cos(pm_rad)
        pm_y = Y_f + r_vis * np.sin(pm_rad)
        prime_meridian.set_data([X_f, pm_x], [Y_f, pm_y])

        # 3. 太陽直下点 (水色点)
        # 幾何学的には常に太陽方向だが、データ検証のため SPICEのSSL値を使って計算して描画
        # PM角度 + SSL = 太陽直下点の角度
        ssl_mark_rad = pm_rad + np.radians(ssl_f)
        ssl_x = X_f + r_vis * np.cos(ssl_mark_rad)
        ssl_y = Y_f + r_vis * np.sin(ssl_mark_rad)
        ssl_marker.set_data([ssl_x], [ssl_y])

        # 4. グラフ更新
        current_time_bar.set_xdata([frame])

        # 5. テキスト
        info_text.set_text(f'Step: {frame}\n'
                           f'Dist: {r_au[frame]:.4f} AU\n'
                           f'Apparent Speed: {curr_app_w:.2f} deg/day\n'
                           f'Mode: {status_str}\n'
                           f'SSL: {ssl_f:.2f} deg')

        return mercury_body, prime_meridian, ssl_marker, current_time_bar, info_text

    # --- 実行 ---
    skip = 10
    frames = np.arange(0, len(data), skip)

    print("Animation generating...")
    ani = FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
    plt.tight_layout()
    plt.show()


# --- メイン処理 ---
if __name__ == '__main__':
    orbit_data = load_orbit_data()
    if orbit_data is not None:
        animate_orbit_check(orbit_data)