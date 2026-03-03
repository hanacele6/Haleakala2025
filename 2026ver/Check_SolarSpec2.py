import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.ndimage import median_filter
from astropy.io import fits

# ---------------------------------------------------------
# 1. パス設定（ここだけ環境に合わせて書き換えてください）
# ---------------------------------------------------------
fits_path = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20250824/master_sky.fits"
solar_spec_path = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/psg/psgrad586-596.txt"
target_fiber = 0

# ---------------------------------------------------------
# 2. 仮説のセットアップ：ここが検証の肝です
# ---------------------------------------------------------
# L1~L3 (確定) + ユーザー仮説 (591.417nm = 1942px)
hypothesis_w = [588.995, 589.300, 589.592, 591.417]
hypothesis_p = [1192,    1280,    1374,    1942]

print(f"検証する仮説ペア数: {len(hypothesis_w)}")
for w, p in zip(hypothesis_w, hypothesis_p):
    print(f"  Wave: {w:.3f} nm <--> Pixel: {p}")

# ---------------------------------------------------------
# 3. データ読み込み
# ---------------------------------------------------------
def gaussian_kernel(size, sigma=1):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

# 太陽データ
spMdl = np.loadtxt(solar_spec_path, skiprows=14)
wavair2vac = 1.000276
xm = spMdl[:, 0] / wavair2vac
kernel = gaussian_kernel(size=181, sigma=61)
ym = convolve(spMdl[:, 1], kernel, mode='same')
ym = ym / np.median(ym)

# 観測データ
with fits.open(fits_path) as hdul:
    spDat = hdul[0].data
    y_obs_raw = spDat[target_fiber, :]
    nx = len(y_obs_raw)
    x_pix = np.arange(nx)
    # 見やすくするために少しスムーズにする
    y_obs = median_filter(y_obs_raw, size=3)

# ---------------------------------------------------------
# 4. 仮説に基づく波長マッピングを作成
# ---------------------------------------------------------
# 4点あるので2次関数でフィットさせます（1次だと湾曲に対応できないため）
coef = np.polyfit(hypothesis_p, hypothesis_w, 2)
pred_wav = np.polyval(coef, x_pix)

# ---------------------------------------------------------
# 5. 結果の表示（上下を完全に同期）
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
plt.subplots_adjust(hspace=0.05) # グラフの間隔を詰めて比較しやすくする

# 上段：太陽参照スペクトル
ax1.plot(xm, ym, color='tab:blue', label='Solar Reference')
ax1.set_ylabel('Normalized Intensity')
ax1.set_title(f'Hypothesis Verification: Is 591.417nm at 1942px?')
ax1.grid(True, linestyle=':', alpha=0.7)

# 下段：観測データ（横軸を「仮説から計算された波長」に変換済み）
ax2.plot(pred_wav, y_obs, color='black', label='Observed Data (Mapped)')
ax2.set_xlabel('Wavelength (nm) [Calculated from Hypothesis]')
ax2.set_ylabel('Intensity (Counts)')
ax2.grid(True, linestyle=':', alpha=0.7)

# --- 検証用の補助線 ---

# 1. フィットに使った4点（緑色）
for w in hypothesis_w:
    ax1.axvline(w, color='tab:green', linestyle='-', linewidth=2, alpha=0.5)
    ax2.axvline(w, color='tab:green', linestyle='-', linewidth=2, alpha=0.5)

# 2. 仮説確認用のテキスト
ax2.text(hypothesis_w[-1], ax2.get_ylim()[0], "Hypothesis Point", color='tab:green',
         ha='right', va='bottom', rotation=90, fontweight='bold')

# 3. 全体を見るためのガイド（赤点線）
# もし仮説が正しければ、これらの点線上に「使っていないはずの谷」も乗るはず
check_lines = [588.39, 590.56, 591.00] # L0, L4, L5相当
for w in check_lines:
    ax1.axvline(w, color='tab:red', linestyle='--', alpha=0.5)
    ax2.axvline(w, color='tab:red', linestyle='--', alpha=0.5)

# 表示範囲を観測データの範囲に合わせる
ax1.set_xlim(pred_wav[0], pred_wav[-1])

plt.show()