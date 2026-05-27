import numpy as np
import matplotlib.pyplot as plt

# 1. 2次元のピクセルグリッド（画面）を作成
resolution = 500
x = np.linspace(-1.1, 1.1, resolution)
y = np.linspace(-1.1, 1.1, resolution)
X, Y = np.meshgrid(x, y)

# 2. 球体のZ座標と法線ベクトルを計算
# x^2 + y^2 <= 1 の領域が球体
sphere_mask = (X**2 + Y**2) <= 1
Z = np.zeros_like(X)
Z[sphere_mask] = np.sqrt(1 - X[sphere_mask]**2 - Y[sphere_mask]**2)

# 3. mu_0 (N dot L) と mu (N dot V) の計算
# 太陽は右(X方向)から、カメラは正面(Z方向)から
mu_0 = X  # 法線ベクトル(X,Y,Z)と太陽(1,0,0)の内積
mu = Z    # 法線ベクトル(X,Y,Z)とカメラ(0,0,1)の内積

# 光が当たっている（かつカメラから見える）領域のマスク（半月）
illuminated_mask = sphere_mask & (mu_0 > 0)

# 4. 各モデルの反射率計算関数
def calc_lambert(mu_0, mask):
    img = np.zeros_like(mu_0)
    img[mask] = mu_0[mask]
    return img

def calc_minnaert(mu_0, mu, mask, nu=0.6):
    img = np.zeros_like(mu_0)
    mu_safe = np.clip(mu[mask], 1e-5, 1.0) # ゼロ割防止
    img[mask] = (mu_0[mask]**nu) * (mu_safe**(nu - 1))
    return img

def calc_lommel_seeliger(mu_0, mu, mask):
    img = np.zeros_like(mu_0)
    img[mask] = mu_0[mask] / (mu_0[mask] + mu[mask])
    return img

def calc_composite(mu_0, mu, mask, c=0.5):
    img = np.zeros_like(mu_0)
    ls = mu_0[mask] / (mu_0[mask] + mu[mask])
    lam = mu_0[mask]
    img[mask] = 0.5 * ls + c * lam
    return img

# 画像の生成
img_L = calc_lambert(mu_0, illuminated_mask)
img_M = calc_minnaert(mu_0, mu, illuminated_mask)
img_LS = calc_lommel_seeliger(mu_0, mu, illuminated_mask)
img_C = calc_composite(mu_0, mu, illuminated_mask)

# --- 描画 ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("2D Rendering of Half-Moon Phase (Phase Angle = 90°)", fontsize=16)

# 各画像をプロット (cmap='gray' で白黒画像として表示)
axes[0, 0].imshow(img_L, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[0, 0].set_title("1. Lambert (Strong 3D Shading)")

axes[0, 1].imshow(img_M, cmap='gray', origin='lower', vmin=0, vmax=2)
axes[0, 1].set_title(r"2. Minnaert ($\nu=0.6$, Glowing Rim)")

axes[1, 0].imshow(img_LS, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[1, 0].set_title("3. Lommel-Seeliger (Flat / Moon-like)")

axes[1, 1].imshow(img_C, cmap='gray', origin='lower', vmin=0, vmax=1)
axes[1, 1].set_title("4. Composite L-LS (Blended)")

for ax in axes.flat:
    ax.axis('off') # 軸を隠す

plt.tight_layout()
plt.show()