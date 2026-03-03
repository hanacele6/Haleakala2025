import numpy as np
import matplotlib.pyplot as plt


def sample_isotropic_direction(normal_vector):
    # この関数は球面上に均一な点を生成するため、本質的に等方的です
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    if np.dot(vec, normal_vector) < 0:
        vec = -vec
    return vec


N_SAMPLES = 100000
# 法線をZ軸にするとx,y成分からφを計算しやすい
surface_normal = np.array([0.0, 0.0, 1.0])

cos_theta_iso = []
phi_iso_deg = []  # ★方位角φを格納するリスト

for _ in range(N_SAMPLES):
    direction = sample_isotropic_direction(surface_normal)

    # cos(θ)を計算 (これはZ成分そのもの)
    cos_t = direction[2]
    cos_theta_iso.append(cos_t)

    # 方位角φを計算
    # np.arctan2(y, x) を使うと正しい象限の角度が得られる
    phi_rad = np.arctan2(direction[1], direction[0])
    phi_iso_deg.append(np.degrees(phi_rad))

# --- 結果をプロット ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

# 1. cos(θ)のヒストグラム (前回同様、平らになる)

ax1.hist(cos_theta_iso, bins=50, density=True, range=(0, 1))
ax1.set_xlabel('cos(θ)' , fontsize = "12")
ax1.set_ylabel('Probability Density' , fontsize = "12")
ax1.grid(True)

# 2. ★方位角φのヒストグラム (こちらも平らになるはず)
ax2.hist(phi_iso_deg, bins=50, density=True, range=(-180, 180))
ax2.set_xlabel('Azimuthal Angle φ [degrees]' , fontsize = "12")
ax2.set_ylabel('Probability Density' , fontsize = "12")
ax2.grid(True)

plt.show()