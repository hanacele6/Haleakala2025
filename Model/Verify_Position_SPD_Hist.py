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
r0 = 0.4 * AU  # [m] (太陽からの距離)

# Na表面密度
sigma_na_initial = 1.5e23 / (1e3) ** 2  # [atoms/m^2]

# シミュレーションの時間ステップ
dt = 10.0  # [s]

# ==============================================================================
# 2. 水星表面のグリッド設定
# ==============================================================================
N_LAT = 24  # 緯度方向の分割数
N_LON = 48  # 経度方向の分割数

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

        # 太陽は+X方向にあると仮定 (太陽天頂角Z)
        cos_Z = np.cos(lat_center) * np.cos(lon_center)
        if cos_Z <= 0:
            continue  # 夜側なのでスキップ

        # --- 放出原子数を計算 ---
        F_UV = F_UV_1AU * (AU / r0) ** 2 * (100) ** 2 # [photons/m^2/s] に変換
        Q_PSD_m2 = Q_PSD / (100) ** 2 # [m^2] に変換
        R_PSD = F_UV * Q_PSD_m2 * cos_Z * sigma_na
        N_ejected_total = R_PSD * cell_area * dt

        # --- プロットするスーパーパーティクルの数を決定 ---
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

# 水星のワイヤーフレームを作成
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = MERCURY_RADIUS * np.outer(np.cos(u), np.sin(v))
y_sphere = MERCURY_RADIUS * np.outer(np.sin(u), np.sin(v))
z_sphere = MERCURY_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

# データをプロット
ax.scatter(plot_x, plot_y, plot_z, s=2, c='red', alpha=0.7, label='Ejected Na Superparticles')
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2, linewidth=0.5)

# 軸のフォーマッタを設定して半径単位にする
formatter = FuncFormatter(lambda val, pos: f'{val/MERCURY_RADIUS:.1f}')
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.zaxis.set_major_formatter(formatter)

ax.set_xlabel('X [R_mercury]')
ax.set_ylabel('Y [R_mercury]')
ax.set_zlabel('Z [R_mercury]')
ax.set_title(f'3D Distribution of Ejected Na Particles ({dt} sec)')
ax.set_aspect('equal')
ax.view_init(elev=20., azim=30)
plt.legend()
plt.show()

# ==============================================================================
# 5. 分布の検証プロット作成
# ==============================================================================
N_BINS = 30
# 単位半径 (RADIUS=1) の球で考える
RADIUS = 1.0

# --- ステップ1: 各粒子の天頂角を計算 ---
# 太陽は+X方向にあるので、天頂角θは arccos(x/R) で計算できる
plot_x_norm = np.array(plot_x) / MERCURY_RADIUS
# 浮動小数点精度の問題で値が1.0をわずかに超える場合があるため、クリップする
plot_x_norm = np.clip(plot_x_norm, -1.0, 1.0)
theta_rad = np.arccos(plot_x_norm)

# --- ステップ2: 天頂角のヒストグラムを作成 ---
counts, bin_edges_rad = np.histogram(theta_rad, bins=N_BINS, range=(0, np.pi/2))
bin_centers_rad = (bin_edges_rad[:-1] + bin_edges_rad[1:]) / 2

# --- ステップ3: 各ビンの表面積を計算し、強度（粒子数/面積）を求める ---
# 天頂角θ1からθ2までの間の球表面の帯の面積は 2πR^2 * |cos(θ1) - cos(θ2)|
areas = 2 * np.pi * RADIUS**2 * np.abs(np.cos(bin_edges_rad[:-1]) - np.cos(bin_edges_rad[1:]))
intensity = np.divide(counts, areas, where=areas > 1e-9)

# --- ステップ4: 結果をプロットするため、理論値と比較できるように規格化 ---
# 計算された強度の最大値が1になるように規格化
if np.max(intensity) > 0:
    normalized_intensity = intensity / np.max(intensity)
else:
    normalized_intensity = intensity # 強度が全て0の場合

# --- プロット作成 ---
fig_verify, ax_verify = plt.subplots(figsize=(12, 7))
ax_verify.set_title('Verification of Particle Ejection Distribution', fontsize=16)

# 計算結果を棒グラフで表示
ax_verify.bar(bin_centers_rad, normalized_intensity,
       width=(bin_edges_rad[1] - bin_edges_rad[0]),
       label='Simulated Intensity\n(Particles per Unit Area)',
       alpha=0.7, color='coral', edgecolor='black')

# 理論曲線をプロット
x_theory_rad = np.linspace(0, np.pi/2, 200)
y_theory = np.cos(x_theory_rad) # 強度は cos(θ) に比例する
ax_verify.plot(x_theory_rad, y_theory, 'b-', lw=2.5, label='Theoretical Curve: I(θ) ∝ cos(θ)')

ax_verify.set_xlabel('Solar Zenith Angle θ [radians]', fontsize=14)
ax_verify.set_ylabel('Normalized Intensity', fontsize=14)
ax_verify.grid(True, linestyle='--', alpha=0.6)
ax_verify.legend(fontsize=12)
ax_verify.set_xlim(0, np.pi/2)
ax_verify.set_ylim(0, 1.1)
plt.tight_layout()
plt.show()