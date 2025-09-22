import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os

# --- ★★★ 設定項目 ★★★ ---

# 1. 視覚化したいシミュレーション結果の.npyファイルを指定
#    特定のTAAのファイルパスをここに記述してください
FILE_TO_LOAD = r"C:\Users\hanac\University\Senior\Mercury\Haleakala2025\SimulationResult3D\density3d_beta0.50_Q2.0_MW_ISO_PD_pl24x24\density3d_taa10_beta0.50_Q2.0_MW_ISO_PD_pl24x24.npy"

# 2. シミュレーションで使用したグリッド設定を以下に記述
#    (シミュレーション本体のmain()関数内の設定と一致させてください)
N_R = 100  # 半径方向の分割数
N_THETA = 24  # 極角(緯度方向)の分割数
N_PHI = 24  # 方位角(経度方向)の分割数
GRID_RADIUS_RM = 5.0  # グリッドの最大半径（水星半径単位）


# --- ここまで設定項目 ---


def visualize_density(filepath, n_r, n_theta, n_phi, grid_radius_rm):
    """
    3D数密度データを読み込み、XY平面とXZ平面の断面図を生成・保存する。
    """
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません: {filepath}")
        return

    # --- 1. データの読み込みと準備 ---
    print(f"ファイルを読み込み中: {filepath}")
    density_cm3 = np.load(filepath)

    # 密度が0以下の値を持つ場合があるため、非常に小さな値を加えておく (対数プロットのため)
    density_cm3[density_cm3 <= 0] = 1e-10

    # --- 2. プロット用の座標グリッドを作成 ---
    # 各セルの中心における半径、極角、方位角の値を計算
    r_vals = np.linspace(0, grid_radius_rm, n_r)
    theta_vals = np.linspace(0, np.pi, n_theta)  # 極角: 0(北極)からpi(南極)
    phi_vals = np.linspace(-np.pi, np.pi, n_phi)  # 方位角: -piからpi

    # 3Dグリッドを作成
    r, theta, phi = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')

    # 球座標をデカルト座標に変換
    # X: 太陽方向, Y: 朝側, Z: 北極方向
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # --- 3. XY平面（赤道断面図）のプロット ---
    print("XY平面（赤道断面図）を作成中...")

    # 赤道に最も近いthetaのインデックスを見つける (theta = pi/2)
    itheta_eq = np.abs(theta_vals - np.pi / 2.0).argmin()

    # データをスライス
    x_slice_xy = x[:, itheta_eq, :]
    y_slice_xy = y[:, itheta_eq, :]
    density_slice_xy = density_cm3[:, itheta_eq, :]

    fig, ax = plt.subplots(figsize=(10, 8))

    # pcolormeshを使用してプロット
    pcm = ax.pcolormesh(
        x_slice_xy, y_slice_xy, density_slice_xy,
        norm=LogNorm(), cmap='plasma', shading='auto'
    )

    # 水星本体を円で表示
    mercury_circle = plt.Circle((0, 0), 1, color='black', fill=True)
    ax.add_artist(mercury_circle)

    # 太陽方向を示す矢印
    ax.arrow(grid_radius_rm * 0.7, 0, grid_radius_rm * 0.2, 0,
             head_width=0.2, head_length=0.2, fc='yellow', ec='black')
    ax.text(grid_radius_rm * 0.7, -0.3, 'To Sun', color='yellow', ha='center')

    ax.set_title(f'Equatorial Plane (XY) Slice\n{os.path.basename(filepath)}', fontsize=16)
    ax.set_xlabel('X axis ($R_M$) - Sun Direction', fontsize=12)
    ax.set_ylabel('Y axis ($R_M$) - Dawn Direction', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-grid_radius_rm, grid_radius_rm)
    ax.set_ylim(-grid_radius_rm, grid_radius_rm)
    ax.grid(True, linestyle='--', alpha=0.5)

    # カラーバーを追加
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Number Density (atoms/cm³)', fontsize=12)

    plt.tight_layout()
    output_filename_xy = os.path.splitext(filepath)[0] + '_XY_slice.png'
    plt.savefig(output_filename_xy)
    print(f"保存しました: {output_filename_xy}")
    plt.show()
    plt.close(fig)

    # --- 4. XZ平面（極断面図）のプロット ---
    print("XZ平面（極断面図）を作成中...")

    # 太陽方向(phi=0)と反太陽方向(phi=pi)に最も近いphiのインデックスを見つける
    iphi_sun = np.abs(phi_vals - 0).argmin()
    iphi_anti = np.abs(phi_vals - np.pi).argmin()  # phi_valsは-piからpiなので、piに最も近いものを探す
    # もしphi_valsが0から2piで定義されていたら、-piはpiと同じインデックスになる場合があるので注意
    if iphi_anti >= len(phi_vals): iphi_anti = np.abs(phi_vals - (-np.pi)).argmin()

    # データをスライスして結合
    x_slice_xz = np.hstack((x[:, :, iphi_anti], x[:, :, iphi_sun]))
    z_slice_xz = np.hstack((z[:, :, iphi_anti], z[:, :, iphi_sun]))
    density_slice_xz = np.hstack((density_cm3[:, :, iphi_anti], density_cm3[:, :, iphi_sun]))

    fig, ax = plt.subplots(figsize=(10, 8))
    pcm = ax.pcolormesh(
        x_slice_xz, z_slice_xz, density_slice_xz,
        norm=LogNorm(), cmap='plasma', shading='auto'
    )

    mercury_circle = plt.Circle((0, 0), 1, color='black', fill=True)
    ax.add_artist(mercury_circle)

    ax.arrow(grid_radius_rm * 0.7, 0, grid_radius_rm * 0.2, 0,
             head_width=0.2, head_length=0.2, fc='yellow', ec='black')
    ax.text(grid_radius_rm * 0.7, -0.3, 'To Sun', color='yellow', ha='center')

    ax.set_title(f'Polar Plane (XZ) Slice\n{os.path.basename(filepath)}', fontsize=16)
    ax.set_xlabel('X axis ($R_M$) - Sun Direction', fontsize=12)
    ax.set_ylabel('Z axis ($R_M$) - North Direction', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-grid_radius_rm, grid_radius_rm)
    ax.set_ylim(-grid_radius_rm, grid_radius_rm)
    ax.grid(True, linestyle='--', alpha=0.5)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Number Density (atoms/cm³)', fontsize=12)

    plt.tight_layout()
    output_filename_xz = os.path.splitext(filepath)[0] + '_XZ_slice.png'
    plt.savefig(output_filename_xz)
    print(f"保存しました: {output_filename_xz}")
    plt.show()
    plt.close(fig)

    print("\n可視化が完了しました。")


if __name__ == '__main__':
    visualize_density(FILE_TO_LOAD, N_R, N_THETA, N_PHI, GRID_RADIUS_RM)