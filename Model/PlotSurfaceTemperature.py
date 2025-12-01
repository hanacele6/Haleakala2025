# -*- coding: utf-8 -*-
"""
水星表面温度 3Dプロット スクリプト

指定した 真近点離角 (TAA) における水星の太陽距離 (AU) を
軌道ファイルから取得し、Leblanc (2003) のモデルに基づいて
表面温度分布を計算し、3D球体にプロットします。

必要なファイル:
1. orbit2025_v5.txt (TAA, AU の取得のため)

必要なライブラリ:
- numpy
- matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# ==============================================================================
# 物理定数 (シミュレーションコードから抜粋)
# ==============================================================================
PHYSICAL_CONSTANTS = {
    'PI': np.pi,
    'AU': 1.496e11,  # [m]
    'RM': 2.440e6,  # [m]
}


# ==============================================================================
# 温度計算モデル (Leblanc 2003, )
# ==============================================================================

def calculate_surface_temperature_leblanc(lon_fixed_rad, lat_rad, AU, subsolar_lon_rad):
    """
    水星表面の局所的な温度を計算します (シミュレーションコードのモデル)。

    Args:
        lon_fixed_rad (float): 計算対象地点の経度 (惑星固定座標系) [rad]
        lat_rad (float): 計算対象地点の緯度 (惑星固定座標系) [rad]
        AU (float): 現在の太陽からの距離 [天文単位]
        subsolar_lon_rad (float): 太陽直下点の経度 (惑星固定座標系) [rad]

    Returns:
        float: 表面温度 [K]
    """
    T_night = 100.0  # 夜側の最低温度 [K]

    # 太陽天頂角の余弦 (cos(SZA)) を計算
    cos_theta = np.cos(lat_rad) * np.cos(lon_fixed_rad - 0)

    if cos_theta <= 0:
        return T_night  # 夜側

    # --- ここからがシミュレーションコードの計算式 ---

    # 論文の T0, T1 とは異なる、シミュレーションコード内の係数設定
    T0 = 100.0  # (シミュレーションコードでの T0)
    T1 = 600.0  # (シミュレーションコードでの T1)

    # 要求された (0.306 / AU)**2 のスケーリングファクター
    scaling_factor = (0.306 / AU) ** 2

    # シミュレーションコードの計算式
    return T0 + T1 * (cos_theta ** 0.25) * scaling_factor


# ==============================================================================
# ヘルパー関数
# ==============================================================================

def get_au_for_taa(taa_target_deg, orbit_data):
    """
    軌道データから、指定したTAAに最も近いAU値を取得します。

    Args:
        taa_target_deg (float): 目標のTAA (度)
        orbit_data (np.ndarray): 読み込んだ軌道データ

    Returns:
        float: 対応するAU値
    """
    taa_col = orbit_data[:, 0]  # TAA (deg)
    au_col = orbit_data[:, 1]  # AU

    # taa_target_deg に最も近いTAAのインデックスを見つける
    idx_closest = np.argmin(np.abs(taa_col - taa_target_deg))

    return au_col[idx_closest]


# ==============================================================================
# メイン実行部
# ==============================================================================
if __name__ == '__main__':

    # --- 1. 設定 ---
    # ここでTAAを指定してください
    TAA_TARGET_DEG = 0

    ORBIT_FILE = 'orbit2025_v6.txt'

    # --- 2. 軌道ファイルの読み込み ---
    try:
        orbit_data = np.loadtxt(ORBIT_FILE)
    except FileNotFoundError:
        print(f"エラー: 軌道ファイル '{ORBIT_FILE}' が見つかりません。")
        print("スクリプトと同じディレクトリに配置してください。")
        sys.exit()
    except Exception as e:
        print(f"エラー: 軌道ファイルの読み込み中に問題が発生しました。 {e}")
        sys.exit()

    # --- 3. AUの取得 ---
    current_au = get_au_for_taa(TAA_TARGET_DEG, orbit_data)
    print(f"指定された TAA = {TAA_TARGET_DEG}°")
    print(f"最も近い軌道データから AU = {current_au:.4f} を取得しました。")

    # --- 4. プロット用データの生成 ---

    # 惑星固定座標系での太陽直下点経度
    # (プロットのため、太陽直下点を経度 0 度に固定)
    SUBSOLAR_LON_RAD = 180.0


    # 球体プロット用の経度・緯度グリッドを作成
    # 経度 (惑星固定座標系, -180° ～ +180°)
    lon = np.linspace(-np.pi, np.pi, 100)
    # 緯度 (-90° ～ +90°)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 50)

    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # 各グリッド点での温度を計算
    # T_grid は (latの数, lonの数) の形状になる
    T_grid = np.zeros_like(lon_grid)
    for i in range(lat_grid.shape[0]):
        for j in range(lon_grid.shape[1]):
            T_grid[i, j] = calculate_surface_temperature_leblanc(
                lon_grid[i, j],
                lat_grid[i, j],
                current_au,
                SUBSOLAR_LON_RAD
            )

    # 3Dプロット用に球面座標を直交座標(X, Y, Z)に変換
    # 座標系の定義:
    # +X: 太陽方向 (経度 0)
    # +Y: 公転進行方向 (経度 +90°)
    # +Z: 北極
    X = PHYSICAL_CONSTANTS['RM'] * np.cos(lat_grid) * np.cos(lon_grid)
    Y = PHYSICAL_CONSTANTS['RM'] * np.cos(lat_grid) * np.sin(lon_grid)
    Z = PHYSICAL_CONSTANTS['RM'] * np.sin(lat_grid)

    # --- 5. 3Dプロットの実行 ---

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 温度データを正規化 (0.0 ～ 1.0) して色にマッピング
    norm = plt.Normalize(vmin=0, vmax=700.0)
    colors = plt.cm.hot(norm(T_grid))

    # 3D表面プロット
    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=colors,
        linewidth=0, antialiased=False, shade=False
    )

    # カラーバーの設定
    m = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=norm)
    m.set_array(T_grid)
    fig.colorbar(m,ax=ax, shrink=0.5, aspect=10, label='Temperature (K)')

    # 軸の設定
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(f'Mercury Surface Temperature at TAA = {TAA_TARGET_DEG}° (AU = {current_au:.3f})')

    # 軸のスケールを均等にする
    max_range = PHYSICAL_CONSTANTS['RM']
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1, 1, 1])  # アスペクト比を1:1:1に

    # 視点の調整 (太陽直下点が見やすいように)
    ax.view_init(elev=30., azim=30)

    plt.show()