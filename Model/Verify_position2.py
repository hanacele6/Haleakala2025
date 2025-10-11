import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PI = np.pi
# サンプル数を減らして描画を高速化
N_SAMPLES = 5000

x_samples = []
y_samples = []
z_samples = []

for _ in range(N_SAMPLES):
    # 経度方向(phi)を日照側(-pi/2 から +pi/2)でサンプリング
    phi_source = PI * np.random.random() - (PI / 2.0)
    # 緯度方向を示すcos(theta)は全球(-1から1)でサンプリング
    cos_theta_source = 2 * np.random.random() - 1.0

    # sin(theta)を計算
    # sin^2(theta) + cos^2(theta) = 1 より
    sin_theta_source = np.sqrt(1.0 - cos_theta_source ** 2)

    # 球面座標系から3D直交座標系へ変換 (半径 R=1 とする)
    x = sin_theta_source * np.cos(phi_source)
    y = sin_theta_source * np.sin(phi_source)
    z = cos_theta_source

    x_samples.append(x)
    y_samples.append(y)
    z_samples.append(z)

# --- 3Dでプロット ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# サンプルした点をプロット
ax.scatter(x_samples, y_samples, z_samples, s=5, c='blue', alpha=0.6, label='Sampled Points')

# 参考のために全球のワイヤーフレームを描画
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2, linewidth=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Distribution of Sampled Points on a Hemisphere')
ax.set_aspect('equal')  # アスペクト比を揃える
plt.legend()
plt.show()