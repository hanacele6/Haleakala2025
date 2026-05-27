# -*- coding: utf-8 -*-
"""
check_orbit_plot_v7.py
==============================================================================
generate_orbit_spice.pyで生成された水星軌道データ
(orbit2025_spice.txt)を視覚的にチェックするスクリプト。

【今回の修正】
- GIF保存が長すぎる問題を解決。
- フレームのスキップ数を増やし、最大フレーム数（150コマ）の制限を追加して高速化。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

# 定数
MERCURY_RADIUS_AU = 0.05
AU_TO_KM = 1.496e8
MERCURY_SPIN_PERIOD_DAY = 58.646


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
    taa_deg = data[:, 0]
    r_au = data[:, 1]
    vt_ms = data[:, 4]
    ssl_deg = data[:, 5]

    # --- 物理量の計算 ---
    r_km = r_au * AU_TO_KM
    vt_km_s = vt_ms / 1000.0
    omega_orb_rad_s = vt_km_s / r_km
    omega_orb_deg_day = np.degrees(omega_orb_rad_s) * (24 * 3600)

    spin_rate_deg_day = 360.0 / MERCURY_SPIN_PERIOD_DAY
    apparent_rotation_speed = spin_rate_deg_day - omega_orb_deg_day

    # --- 描画準備 ---
    taa_rad = np.radians(taa_deg)
    X_merc = r_au * np.cos(taa_rad)
    Y_merc = r_au * np.sin(taa_rad)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1.2, 1.0])
    ax_orbit = fig.add_subplot(gs[0, 0])
    ax_merc = fig.add_subplot(gs[0, 1])
    ax_graph = fig.add_subplot(gs[1, :])

    # ==========================================
    # [左上] 軌道プロット
    # ==========================================
    ax_orbit.set_title("Mercury Orbit System", fontsize=16, fontweight='bold')
    ax_orbit.set_xlabel("X Position [AU]", fontsize=14)
    ax_orbit.set_ylabel("Y Position [AU]", fontsize=14)
    ax_orbit.tick_params(axis='both', labelsize=12)
    ax_orbit.set_aspect('equal', adjustable='box')
    max_r = np.max(r_au) * 1.1
    ax_orbit.set_xlim(-max_r, max_r)
    ax_orbit.set_ylim(-max_r, max_r)

    ax_orbit.plot(0, 0, 'o', color='yellow', markersize=20, label='Sun')
    ax_orbit.plot(X_merc, Y_merc, ':', color='gray', alpha=0.4, linewidth=0.5)

    mercury_body, = ax_orbit.plot([], [], 'o', color='lightgray', markersize=25, zorder=5)
    prime_meridian, = ax_orbit.plot([], [], '-', color='red', linewidth=2, zorder=6)
    ssl_marker, = ax_orbit.plot([], [], 'o', color='cyan', markersize=6, zorder=7)

    taa_text = ax_orbit.text(0.02, 0.98, '', transform=ax_orbit.transAxes, color='white',
                             fontsize=16, fontweight='bold', verticalalignment='top', 
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='white'))
    ax_orbit.legend(loc='upper right', fontsize=12)

    # ==========================================
    # [右上] 水星のアップ図
    # ==========================================
    ax_merc.set_aspect('equal')
    ax_merc.set_xlim(-1.5, 1.5)
    ax_merc.set_ylim(-1.5, 1.5)
    ax_merc.axis('off')
    ax_merc.set_title("Mercury Close-up", fontsize=16, fontweight='bold')

    circle_merc = patches.Circle((0, 0), 1.0, color='lightgray', zorder=1)
    ax_merc.add_patch(circle_merc)

    night_patch = patches.Wedge((0, 0), 1.0, -90, 90, color='black', alpha=0.6, zorder=2)
    ax_merc.add_patch(night_patch)

    ax_merc.annotate('', xy=(-1.0, 0), xytext=(-1.4, 0),
                     arrowprops=dict(facecolor='yellow', edgecolor='yellow', width=2, headwidth=8), zorder=3)
    ax_merc.text(-1.45, 0, 'Sun', color='yellow', ha='right', va='center', fontsize=14, fontweight='bold')
    
    ax_merc.text(0, -1.35, "Red Line: Prime Meridian\n(0° Longitude rotating with spin)", 
                 color='red', ha='center', va='center', fontsize=13, fontweight='bold')

    ax_merc.plot([-1.0], [0.0], 'o', color='cyan', markersize=8, zorder=4)
    pm_line_merc, = ax_merc.plot([], [], '-', color='red', linewidth=3, zorder=5)

    # ==========================================
    # [下段] グラフ
    # ==========================================
    ax_graph.set_title("Apparent Rotation Speed vs TAA", fontsize=16, fontweight='bold')
    ax_graph.set_xlabel("True Anomaly (TAA) [deg]", fontsize=14)
    ax_graph.set_ylabel("Speed [deg/day]", fontsize=14)
    ax_graph.tick_params(axis='both', labelsize=12)
    ax_graph.grid(True, linestyle='--', alpha=0.3)
    ax_graph.set_xlim(0, 360)

    taa_mod = taa_deg % 360.0
    sort_idx = np.argsort(taa_mod)
    sorted_taa = taa_mod[sort_idx]
    sorted_speed = apparent_rotation_speed[sort_idx]

    ax_graph.axhline(0, color='white', linewidth=1)
    ax_graph.plot(sorted_taa, sorted_speed, color='cyan')
    ax_graph.fill_between(sorted_taa, sorted_speed, 0,
                          where=(sorted_speed < 0),
                          color='red', alpha=0.4, label='Retrograde (Reverse)')

    current_time_bar = ax_graph.axvline(x=0, color='yellow', linestyle='-', linewidth=2)
    ax_graph.legend(loc='upper right', fontsize=12)

    # ==========================================
    # アニメーション更新関数
    # ==========================================
    def update(frame):
        X_f, Y_f = X_merc[frame], Y_merc[frame]
        taa_f = taa_deg[frame]
        ssl_f = ssl_deg[frame]
        curr_app_w = apparent_rotation_speed[frame]
        taa_mod_f = taa_mod[frame]

        merc_color = 'salmon' if curr_app_w < 0 else 'lightgray'
        
        mercury_body.set_data([X_f], [Y_f])
        mercury_body.set_color(merc_color)

        pm_angle_deg = (taa_f + 180.0 - ssl_f) % 360.0
        pm_rad = np.radians(pm_angle_deg)
        r_vis = MERCURY_RADIUS_AU * 0.8
        pm_x = X_f + r_vis * np.cos(pm_rad)
        pm_y = Y_f + r_vis * np.sin(pm_rad)
        prime_meridian.set_data([X_f, pm_x], [Y_f, pm_y])

        ssl_mark_rad = pm_rad + np.radians(ssl_f)
        ssl_x = X_f + r_vis * np.cos(ssl_mark_rad)
        ssl_y = Y_f + r_vis * np.sin(ssl_mark_rad)
        ssl_marker.set_data([ssl_x], [ssl_y])

        pm_angle_merc = 180.0 - ssl_f
        pm_rad_merc = np.radians(pm_angle_merc)
        pm_line_merc.set_data([0, np.cos(pm_rad_merc)], [0, np.sin(pm_rad_merc)])

        current_time_bar.set_xdata([taa_mod_f, taa_mod_f])
        taa_text.set_text(f'TAA = {taa_mod_f:05.1f}°')

        return mercury_body, prime_meridian, ssl_marker, pm_line_merc, current_time_bar, taa_text

    # --- 実行とGIF保存 ---
    skip = 25        
    max_frames = 1200 
    
    total_possible_frames = len(data) // skip
    frames_to_render = min(max_frames, total_possible_frames)
    
    # 実際に処理するフレームのインデックス配列を作成
    frames = np.arange(0, frames_to_render * skip, skip)

    print(f"Generating Animation... Saving {frames_to_render} frames as GIF.")
    
    ani = FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
    plt.tight_layout()
    
    gif_filename = 'mercury_orbit_check.gif'
    ani.save(gif_filename, writer=PillowWriter(fps=15))
    print(f"Success! Saved as '{gif_filename}'")
    
    plt.show()


# --- メイン処理 ---
if __name__ == '__main__':
    orbit_data = load_orbit_data()
    if orbit_data is not None:
        animate_orbit_check(orbit_data)