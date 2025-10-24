import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



def plot_spherical_region(ax, theta_range_deg, phi_range_deg, color, label):
    """
    3D球体表面に特定の領域を描画する。
    Nightsideのような経度方向(phi)にまたがる領域も扱えるようにする。
    """
    # 角度の範囲を度からラジアンに変換
    theta_min, theta_max = np.deg2rad(theta_range_deg)
    phi_min, phi_max = np.deg2rad(phi_range_deg)

    radius = 1.0  # 可視化には単位球を使用

    # 指定されたphiの範囲で曲面を生成する関数
    def generate_surface(p_min, p_max):
        theta = np.linspace(theta_min, theta_max, 50)
        phi = np.linspace(p_min, p_max, 50)
        theta, phi = np.meshgrid(theta, phi)

        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)
        return x, y, z

    # 通常の場合と、経度が-180/180度をまたぐ場合（Nightsideなど）の処理
    if phi_min <= phi_max:
        x, y, z = generate_surface(phi_min, phi_max)
        ax.plot_surface(x, y, z, color=color, alpha=0.7, rstride=2, cstride=2, label=label)
    else:
        # 経度180度をまたぐ領域の処理
        # 1. phi_min から 180度(pi) までを描画
        x1, y1, z1 = generate_surface(phi_min, np.pi)
        ax.plot_surface(x1, y1, z1, color=color, alpha=0.7, rstride=2, cstride=2)

        # 2. -180度(-pi) から phi_max までを描画
        x2, y2, z2 = generate_surface(-np.pi, phi_max)
        ax.plot_surface(x2, y2, z2, color=color, alpha=0.7, rstride=2, cstride=2)


# --- ★★★ 解析領域の定義（元のスクリプトからコピー） ★★★ ---
REGIONS_TO_ANALYZE = [
    {'label': 'Dayside', 'theta_range_deg': (1, 179), 'phi_range_deg': (-89, 89)},
    {'label': 'Dawnside', 'theta_range_deg': (10, 170), 'phi_range_deg': (10, 80)},
    {'label': 'Duskside', 'theta_range_deg': (1, 179), 'phi_range_deg': (-89, -1)},
    {'label': 'Nightside', 'theta_range_deg': (30, 150), 'phi_range_deg': (135, -135)},
    {'label': 'Source_Dusk_Terminator', 'theta_range_deg': (10, 170), 'phi_range_deg': (90, 96)},
    {'label': 'Source_Dawn_Terminator', 'theta_range_deg': (10, 170), 'phi_range_deg': (-96, -90)},
]

# --- ★★★ 表示する領域を選択 ★★★ ---
# 複数の領域を同時に表示することも可能です。例: ['Dayside', 'Nightside']
LABELS_TO_PLOT = ['Duskside']
# 全ての領域をプロットする場合: LABELS_TO_PLOT = [r['label'] for r in REGIONS_TO_ANALYZE]


# --- メインの描画処理 ---
if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 半透明の基準球体を描画
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2, rstride=5, cstride=5)

    # 2. 各領域の色を定義し、選択された領域をプロット
    colors = {'Dayside': 'gold', 'Dawnside': 'lightgreen', 'Duskside': 'tomato', 'Nightside': 'darkviolet'}
    legend_patches = []

    regions_to_plot = [r for r in REGIONS_TO_ANALYZE if r['label'] in LABELS_TO_PLOT]

    for region in regions_to_plot:
        label = region['label']
        color = colors.get(label, 'gray')
        print(f"'{label}' を {color}色で描画します...")
        plot_spherical_region(ax, region['theta_range_deg'], region['phi_range_deg'], color, label)
        legend_patches.append(Patch(color=color, label=label))

    # 3. プロットのラベルと外観を設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Analysis Regions', fontsize=16)

    # アスペクト比を固定して真球に見せる
    ax.set_box_aspect([1, 1, 1])

    # 凡例を作成
    ax.legend(handles=legend_patches, loc='upper right')

    plt.show()