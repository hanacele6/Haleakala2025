import numpy as np
import matplotlib.pyplot as plt

PI = np.pi
N_SAMPLES = 100000

phi_samples = []
cos_theta_samples = []

for _ in range(N_SAMPLES):
    # 経度方向(phi)を日照側(-pi/2 から +pi/2)でサンプリング
    phi_source = PI * np.random.random() - (PI / 2.0)
    # 緯度方向を示すcos(theta)は全球(-1から1)でサンプリング
    cos_theta_source = 2 * np.random.random() - 1.0

    phi_samples.append(np.degrees(phi_source))  # 分かりやすいように度に変換
    cos_theta_samples.append(cos_theta_source)

# --- 結果をプロット ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

# 1. 経度(phi)の分布
ax1.hist(phi_samples, bins=50, density=True)

ax1.set_xlabel('Longitude [degrees]' , fontsize = "16")
ax1.set_ylabel('Probability Density' , fontsize = "16")
ax1.grid(True)
ax1.set_xlim(-90, 90)

# 2. cos(緯度)の分布
ax2.hist(cos_theta_samples, bins=50, density=True)

ax2.set_xlabel('cos(theta)' , fontsize = "16")
ax2.set_ylabel('Probability Density' , fontsize = "16")
ax2.grid(True)
ax2.set_xlim(-1, 1)

plt.show()