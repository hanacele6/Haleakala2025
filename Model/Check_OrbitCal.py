# -*- coding: utf-8 -*-
"""
水星軌道データ検証スクリプト
orbit2025_v6.txt の物理的整合性をチェックします。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def verify_orbit_data():
    filename = 'orbit2025_v6.txt'

    if not os.path.exists(filename):
        print(f"エラー: '{filename}' が見つかりません。")
        return

    # データの読み込み
    print(f"Loading {filename} ...")
    data = np.loadtxt(filename)

    # 列の抽出
    # [TAA, AU, Time, Vr, Vt, SubSol]
    r_au = data[:, 1]
    t_sec = data[:, 2]
    v_rad = data[:, 3]
    v_tan = data[:, 4]
    subsol_lon = data[:, 5]

    # --- 定数 (生成スクリプトと同じものを使用) ---
    GM_SUN = 1.32712440018e20
    AU_METER = 1.495978707e11  # 近似値 (比率計算なので厳密でなくてOK)

    # r をメートルに変換 (概算)
    r_meters = r_au * AU_METER

    print("\n" + "=" * 50)
    print(" 1. 保存則のチェック (数値積分の精度)")
    print("=" * 50)

    # 検証1: 角運動量の保存 (L = r * v_tangential)
    # ※ r と v_tan が反比例の関係にあるか
    L_specific = r_meters * v_tan
    L_mean = np.mean(L_specific)
    L_variation = (np.max(L_specific) - np.min(L_specific)) / L_mean * 100

    print(f"角運動量 (L) の変動率: {L_variation:.6f} %")
    if L_variation < 0.01:
        print(">> [合格] 角運動量は非常によく保存されています。")
    else:
        print(">> [警告] 角運動量の変動が大きいです。積分精度を確認してください。")

    # 検証2: 総エネルギーの保存 (E = K + U)
    # K = 0.5 * v^2, U = -GM / r
    v_sq = v_rad ** 2 + v_tan ** 2
    K = 0.5 * v_sq
    U = -GM_SUN / r_meters
    E_total = K + U

    E_mean = np.mean(E_total)
    E_variation = (np.max(E_total) - np.min(E_total)) / np.abs(E_mean) * 100

    print(f"総エネルギー (E) の変動率: {E_variation:.6f} %")
    if E_variation < 0.01:
        print(">> [合格] エネルギー保存則を満たしています。")
    else:
        print(">> [警告] エネルギー変動が大きいです。")

    print("\n" + "=" * 50)
    print(" 2. 3:2 スピン軌道共鳴の視覚的チェック")
    print("=" * 50)
    print("グラフを描画します...")
    print("- 図1: 太陽直下点経度の時間変化 (0日～88日)")
    print("  -> 近日点(t=0, 88付近)でグラフが「平ら」あるいは「波打つ」のが正解です。")
    print("  -> これが「秤動(Libration)」による太陽の逆行/停止現象です。")

    # グラフ描画
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上段: 距離の変化
    ax1.plot(t_sec / (24 * 3600), r_au, label='Distance (AU)', color='blue')
    ax1.set_ylabel('Sun-Mercury Distance [AU]')
    ax1.set_title('Distance from Sun')
    ax1.grid(True)

    # 近日点と遠日点にマーク
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Perihelion')
    ax1.axvline(x=44, color='g', linestyle='--', alpha=0.5, label='Aphelion approx')
    ax1.axvline(x=88, color='r', linestyle='--', alpha=0.5)
    ax1.legend()

    # 下段: 太陽直下点経度
    ax2.plot(t_sec / (24 * 3600), subsol_lon, label='Sub-Solar Longitude', color='orange', linewidth=2)
    ax2.set_ylabel('Longitude [deg]')
    ax2.set_xlabel('Time [days]')
    ax2.set_title('Sub-Solar Point Longitude (Check for Libration at Perihelion)')
    ax2.grid(True)
    ax2.set_ylim(-180, 180)

    # 近日点付近を拡大してみるためのガイド
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=88, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    verify_orbit_data()