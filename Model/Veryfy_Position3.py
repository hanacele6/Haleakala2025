import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter

# ==============================================================================
# 1. 物理定数とシミュレーションパラメータの設定
# ==============================================================================

# 1つの描画点が表現する原子の数 (この値を大きくすると描画点は減る)
SUPERPARTICLE_ATOMS = 1e23

# 光刺激脱離(PSD)に関する定数
F_UV_1AU = 1.5e14  # [photons/cm^2/s]
Q_PSD = 2.0e-20  # [cm^2]

# 水星の物理パラメータ
MERCURY_RADIUS = 2440e3  # [m]
AU = 1.496e11  # [m]
r0 = 0.4 * AU  # [m]

# Na表面密度
sigma_na_initial = 1.5e23 / (1e3) ** 2  # [atoms/m^2]

# シミュレーションの時間ステップ
dt = 10.0  # [s]

# ==============================================================================
# 2. 水星表面のグリッド設定
# ==============================================================================
N_LAT = 24
N_LON = 48

lat = np.linspace(-np.pi / 2, np.pi / 2, N_LAT)
lon = np.linspace(-np.pi, np.pi, N_LON)
dlat = lat[1] - lat[0]
dlon = lon[1] - lon[0]

# ==============================================================================
# 3. 粒子放出位置の生成
# ==============================================================================
plot_x, plot_y, plot_z = [], [], []

# グリッドを一つずつループ
for i in range(N_LAT):
    for j in range(N_LON):
        # --- 各グリッドの中心で物理量を計算 ---
        lat_center = lat[i]
        lon_center = lon[j]

        cell_area = (MERCURY_RADIUS ** 2) * np.cos(lat_center) * dlat * dlon
        sigma_na = sigma_na_initial

        cos_Z = np.cos(lat_center) * np.cos(lon_center)
        if cos_Z <= 0:
            continue  # 夜側なのでスキップ

        # --- 放出原子数を計算 (前のコードと同じ) ---
        F_UV = F_UV_1AU * (AU / r0) ** 2 * 100 ** 2
        Q_PSD_m2 = Q_PSD / 100 ** 2
        R_PSD = F_UV * Q_PSD_m2 * cos_Z * sigma_na
        N_ejected_total = R_PSD * cell_area * dt

        # --- プロットする点の数を決定 ---
        num_points_to_plot = int(np.floor(N_ejected_total / SUPERPARTICLE_ATOMS))

        if num_points_to_plot == 0:
            continue

        # --- 決定した数の点をグリッド内にランダムに配置 ---
        for _ in range(num_points_to_plot):
            # グリッド内でランダムな緯度・経度を生成
            lat_point = lat_center + (np.random.rand() - 0.5) * dlat
            lon_point = lon_center + (np.random.rand() - 0.5) * dlon

            # 球面座標系から直交座標系に変換
            x = MERCURY_RADIUS * np.cos(lat_point) * np.cos(lon_point)
            y = MERCURY_RADIUS * np.cos(lat_point) * np.sin(lon_point)
            z = MERCURY_RADIUS * np.sin(lat_point)

            plot_x.append(x)
            plot_y.append(y)
            plot_z.append(z)

print(f"合計 {len(plot_x):,} 個のスーパーパーティクルを生成しました。")

# ==============================================================================
# 4. 3Dプロットの作成
# ==============================================================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# --- ここからが変更箇所 ---

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = MERCURY_RADIUS * np.outer(np.cos(u), np.sin(v))
y_sphere = MERCURY_RADIUS * np.outer(np.sin(u), np.sin(v))
z_sphere = MERCURY_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

# 1. データを水星半径で割って無次元化する
plot_x_norm = np.array(plot_x) / MERCURY_RADIUS
plot_y_norm = np.array(plot_y) / MERCURY_RADIUS
plot_z_norm = np.array(plot_z) / MERCURY_RADIUS

x_sphere_norm = x_sphere / MERCURY_RADIUS
y_sphere_norm = y_sphere / MERCURY_RADIUS
z_sphere_norm = z_sphere / MERCURY_RADIUS

# ---

# 無次元化したデータでプロット
ax.scatter(plot_x_norm, plot_y_norm, plot_z_norm, s=1, c='red', alpha=0.7, label='Ejected Na Superparticles')
ax.plot_wireframe(x_sphere_norm, y_sphere_norm, z_sphere_norm, color='gray', alpha=0.2, linewidth=0.5)

# 2. 軸ラベルを無次元単位に変更
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 3. カンマ区切りのフォーマッタは不要なので削除

# --- 変更箇所ここまで ---

ax.set_title(f'3D Distribution of Ejected Na Particles ({dt} sec)')
ax.set_aspect('equal')
ax.view_init(elev=20., azim=30)
plt.legend()
plt.show()