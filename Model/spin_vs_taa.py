# -*- coding: utf-8 -*-
"""
plot_apparent_spin_vs_taa.py
==============================================================================
generate_orbit_spice.pyで生成された水星軌道データを使用し、
横軸：TAA (真近点角)、縦軸：見かけの自転速度 (Apparent Rotation Speed)
のグラフを描画するスクリプト。

水星の「3:2 スピン軌道共鳴」により、近日点付近で公転角速度が自転角速度を
上回り、太陽から見て「逆回転（太陽が西から昇る現象）」が起きる様子を可視化します。
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- 定数 (元のスクリプトと同一) ---
AU_TO_KM = 1.496e8  # 1 AU = 1.496e8 km
MERCURY_SPIN_PERIOD_DAY = 58.646  # 水星の自転周期 [day]


def load_orbit_data(filename='orbit2025_spice_unwrapped.txt'):
    """データの読み込み"""
    if not os.path.exists(filename):
        print(f"[Error] '{filename}' が見つかりません。", file=sys.stderr)
        return None
    try:
        data = np.loadtxt(filename, skiprows=1)
        return data
    except Exception as e:
        print(f"[Error] ロード失敗: {e}", file=sys.stderr)
        return None


def plot_analysis(data):
    # --- 1. データ展開 ---
    # カラム: 0:TAA, 1:Dist(AU), 4:Vt(m/s), 5:SSL
    taa_deg = data[:, 0]
    r_au = data[:, 1]
    vt_ms = data[:, 4]

    # --- 2. 物理量の計算 ---
    # A. 公転角速度 (Orbital Angular Velocity)
    r_km = r_au * AU_TO_KM
    vt_km_s = vt_ms / 1000.0
    omega_orb_rad_s = vt_km_s / r_km
    omega_orb_deg_day = np.degrees(omega_orb_rad_s) * (24 * 3600)

    # B. 自転角速度 (Spin Angular Velocity) - 一定
    spin_rate_deg_day = 360.0 / MERCURY_SPIN_PERIOD_DAY

    # C. 見かけの回転速度 (Apparent Rotation Speed)
    # 正 = 順行 (Prograde), 負 = 逆行 (Retrograde)
    apparent_speed = spin_rate_deg_day - omega_orb_deg_day

    # --- 3. グラフ描画用にデータをTAA順にソート ---
    # 時系列データだとTAAが360->0にジャンプする箇所があるため、
    # きれいな曲線を描くためにTAAでソートします。
    sort_indices = np.argsort(taa_deg)
    taa_sorted = taa_deg[sort_indices]
    apparent_sorted = apparent_speed[sort_indices]

    # --- 4. プロット作成 ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # メインの曲線
    ax.plot(taa_sorted, apparent_sorted, color='blue', label='Apparent Rotation Speed')

    # 基準線 (0 deg/day)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')

    # 領域の塗りつぶし
    # 逆行 (Retrograde): 0未満
    ax.fill_between(taa_sorted, apparent_sorted, 0,
                    where=(apparent_sorted < 0),
                    color='red', alpha=0.3, label='Retrograde (Sun moves East)')

    # 順行 (Prograde): 0以上
    ax.fill_between(taa_sorted, apparent_sorted, 0,
                    where=(apparent_sorted > 0),
                    color='cyan', alpha=0.3, label='Prograde (Sun moves West)')

    # --- 補助線と装飾 ---
    # 近日点 (Perihelion)
    #ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    #ax.text(0, np.max(apparent_sorted) * 1.05, 'Perihelion (0°)',
    #        ha='center', color='black', fontweight='bold')

    # 遠日点 (Aphelion)
    #ax.axvline(180, color='gray', linestyle='--', alpha=0.7)
    #ax.text(180, np.max(apparent_sorted) * 1.05, 'Aphelion (180°)',
    #        ha='center', color='black', fontweight='bold')

    # ラベル設定
    #ax.set_title("Mercury's Apparent Rotation Speed vs. True Anomaly (TAA)", fontsize=14)
    ax.set_xlabel("True Anomaly Angle[deg]", fontsize=12)
    ax.set_ylabel("Apparent Speed [deg/day]", fontsize=12)

    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 60))

    # Y軸の見やすさ調整（少し余裕を持たせる）
    y_min, y_max = np.min(apparent_sorted), np.max(apparent_sorted)
    margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - margin, y_max + margin * 2)  # 上部はテキスト用に広めに

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    orbit_data = load_orbit_data()
    if orbit_data is not None:
        plot_analysis(orbit_data)