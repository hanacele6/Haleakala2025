# -*- coding: utf-8 -*-
"""
==============================================================================
解析ツール: [時間比較] 光電離寿命 vs TAA進行時間
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 物理定数
# ==============================================================================
AU_M = 1.496e11  # 1 AU [m]
GM_SUN = 1.3271244e20  # 太陽の重力定数 [m^3/s^2]
MERCURY_A_AU = 0.387098  # 水星軌道長半径 [AU]
MERCURY_E = 0.205630  # 水星離心率

# 1AUでの光電離寿命 [秒] (約15.14時間)
TAU_1AU = 54500.0


# ==============================================================================
# 計算関数
# ==============================================================================
def get_orbit_time_scales(taa_deg):
    """
    TAAを与えて、以下の2つの「時間」を計算する
    1. 光電離寿命 [hours]
    2. TAAが1度進むのにかかる時間 [hours]
    """
    rad = np.deg2rad(taa_deg)

    # 1. 距離 r [m]
    a_m = MERCURY_A_AU * AU_M
    r_m = a_m * (1 - MERCURY_E ** 2) / (1 + MERCURY_E * np.cos(rad))
    r_au = r_m / AU_M

    # --- A. 光電離寿命 (Survival Time) ---
    # tau = tau_1au * r^2
    tau_s = TAU_1AU * (r_au ** 2)
    lifetime_h = tau_s / 3600.0

    # --- B. 軌道進行時間 (Dynamics Time) ---
    # 角速度 omega = h / r^2 [rad/s]
    h = np.sqrt(GM_SUN * a_m * (1 - MERCURY_E ** 2))
    omega_rad_s = h / (r_m ** 2)
    omega_deg_s = np.rad2deg(omega_rad_s)

    # 1度進むのにかかる秒数 = 1 / omega
    time_per_deg_s = 1.0 / omega_deg_s
    time_per_deg_h = time_per_deg_s / 3600.0

    return lifetime_h, time_per_deg_h


# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    taa_list = np.arange(0, 361, 1)

    lifetime_list = []
    orbit_move_list = []

    for taa in taa_list:
        life, move = get_orbit_time_scales(taa)
        lifetime_list.append(life)
        orbit_move_list.append(move)

    # ==========================================================================
    # プロット: 縦軸を「時間 [Hours]」に統一
    # ==========================================================================
    plt.figure(figsize=(10, 6))

    # 1. 光電離寿命 (青)
    plt.plot(taa_list, lifetime_list, color='blue', linewidth=3, label='Photo-ionization Lifetime (Survival)')

    # 2. TAA 1度進行時間 (赤)
    plt.plot(taa_list, orbit_move_list, color='red', linewidth=3, linestyle='--', label='Time to Advance 1 deg TAA')

    # 数値の注釈 (近日点)
    min_life = min(lifetime_list)
    min_move = min(orbit_move_list)
    plt.text(0, min_life + 0.2, f"Life: {min_life:.1f} h", color='blue', fontweight='bold', ha='center')
    plt.text(0, min_move - 0.5, f"Move 1deg: {min_move:.1f} h", color='red', fontweight='bold', ha='center')

    # 数値の注釈 (遠日点)
    max_life = max(lifetime_list)
    max_move = max(orbit_move_list)
    plt.text(180, max_life - 0.5, f"Life: {max_life:.1f} h", color='blue', fontweight='bold', ha='center')
    plt.text(180, max_move + 0.2, f"Move 1deg: {max_move:.1f} h", color='red', fontweight='bold', ha='center')

    plt.xlabel('True Anomaly Angle (TAA) [deg]', fontsize=14)
    plt.ylabel('Time Scale [Hours]', fontsize=14)
    plt.title('Timescale Comparison: Survival vs. Orbital Motion', fontsize=16)

    plt.xlim(0, 360)
    # 見やすくするためにY軸の下限を0にする
    plt.ylim(0, max(orbit_move_list) * 1.1)

    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=12, loc='upper center')

    plt.tight_layout()
    plt.savefig('timescale_comparison.png', dpi=300)
    print("Saved: timescale_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()