import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ設定 (自由に変更して試せます) ---
# 光学系の設定
d = 1.0 / 600 * 1e6  # 回折格子の溝の間隔 (nm/line)。600本/mmの場合
m = 1  # 次数
alpha_deg = 90  # 回折格子への中心入射角 (度)
f_col = 200  # コリメータの焦点距離 (mm)
f_cam = 100  # カメラレンズの焦点距離 (mm)

# スリット/ファイバーアレイの設定
# yはスリットの高さ方向の位置を表す
y_values = np.linspace(-15, 15, 101)  # スリットの高さ(空間方向)を-15mmから+15mmまでシミュレート

# シミュレーションする波長 (nm)
wavelengths = [450, 550, 650]  # 青、緑、赤の光
colors = ['blue', 'green', 'red']

# --- シミュレーション実行 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 7))

# 基準となる中心位置を計算 (グラフを中央に寄せるため)
alpha_rad_center = np.deg2rad(alpha_deg)
sin_beta_center = (m * wavelengths[1] / d) - np.sin(alpha_rad_center)
beta_center = np.arcsin(sin_beta_center)
x_center = f_cam * np.tan(beta_center)


for wl, color in zip(wavelengths, colors):
    # 1. スリットの高さ(y)から入射角(alpha)を計算
    alpha_rad = np.deg2rad(alpha_deg) + np.arctan(y_values / f_col)

    # 2. 回折格子の式から回折角(beta)を計算
    sin_beta = (m * wl / d) - np.sin(alpha_rad)
    beta_rad = np.arcsin(sin_beta)

    # 3. 回折角(beta)から検出器上の位置(x)を計算
    x_position = f_cam * np.tan(beta_rad) - x_center

    # 4. プロット
    ax.plot(x_position, y_values, label=f'{wl} nm', color=color)

# --- グラフの装飾 ---
ax.set_title('Smile distortion simulation', fontsize=16)
ax.set_xlabel('Horizontal position of detector [misalignment]', fontsize=12)
ax.set_ylabel('Slit/fiber height position (spatial direction) [mm]', fontsize=12)
ax.legend(title='Wavelength')
ax.grid(True)
ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7) # 中心線
plt.axis('equal') # 歪みが分かりやすいようにアスペクト比を1:1に

# グラフを表示
plt.show()