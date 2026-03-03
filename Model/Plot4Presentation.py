import matplotlib.pyplot as plt
import numpy as np


def plot_solid_sphere(lat_div, lon_div):
    # --- パラメータ設定 ---
    r = 1.0

    # 緯度方向の分割 (0 から pi まで)
    phi = np.linspace(0, np.pi, lat_div + 1)

    # 経度方向の分割 (0 から 2pi まで)
    theta = np.linspace(0, 2 * np.pi, lon_div + 1)

    # メッシュグリッドの作成
    PHI, THETA = np.meshgrid(phi, theta)

    # --- 座標変換 ---
    x = r * np.sin(PHI) * np.cos(THETA)
    y = r * np.sin(PHI) * np.sin(THETA)
    z = r * np.cos(PHI)

    # --- プロット設定 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 表面を描画 (plot_surface)
    # color: 面の色（ここではシアン）
    # edgecolor: 分割線の色（ここでは黒くはっきりと表示）
    # linewidth: 分割線の太さ
    # rstride=1, cstride=1: 全てのグリッドを描画する設定
    ax.plot_surface(x, y, z,
                    color='cyan',
                    edgecolor='black',
                    linewidth=0.3,
                    rstride=1, cstride=1,
                    alpha=1.0,  # 不透明度 (1.0で完全に不透明)
                    shade=True)  # 陰影をつける

    # アスペクト比を揃える
    ax.set_box_aspect([1, 1, 1])

    # 軸ラベルなどを非表示にする（球体だけを綺麗に見せるため）
    ax.set_axis_off()

    ax.set_title(f'Solid Sphere with Grid\nLatitude: {lat_div}, Longitude: {lon_div}')

    plt.show()


if __name__ == "__main__":
    # 緯度72分割、経度36分割を指定
    plot_solid_sphere(lat_div=36, lon_div=72)