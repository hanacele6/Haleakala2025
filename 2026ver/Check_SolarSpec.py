import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from astropy.io import fits

# ---------------------------------------------------------
# 1. パスと現在のパラメータ設定（環境に合わせて変更してください）
# ---------------------------------------------------------
fits_path = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20250824/master_sky.fits"  # 観測データのFITSファイルパス
solar_spec_path = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/psg/psgrad586-596.txt"  # 参照太陽スペクトルパス

# 現在のパラメータ
wlinesM = [588.39, 588.995, 589.3, 589.592, 590.560, 591.002, 591.417, 591.63]
pxlinesD0_base = [1012, 1192, 1280, 1374, 1670, 1806, 1942, 2003]

# 確認に使うファイバーのインデックス（通常は中央付近のファイバーが見やすいです）
target_fiber = 0


# ---------------------------------------------------------
# 2. データの読み込みと前処理
# ---------------------------------------------------------
def gaussian_kernel(size, sigma=1):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


# 太陽スペクトル（本番コードと同じぼかし処理を適用）
spMdl = np.loadtxt(solar_spec_path, skiprows=14)
wavair2vac = 1.000276
xm = spMdl[:, 0] / wavair2vac
kernel = gaussian_kernel(size=181, sigma=61)
cv = convolve(spMdl[:, 1], kernel, mode='same')
ym = cv / np.median(cv)

# 観測データ
with fits.open(fits_path) as hdul:
    spDat = hdul[0].data
    ny, nx = spDat.shape
    if target_fiber >= ny:
        target_fiber = ny // 2
    y_obs = spDat[target_fiber, :]
    x_obs = np.arange(nx)

# ---------------------------------------------------------
# 3. 近似波長の計算（上下グラフの表示同期用）
# ---------------------------------------------------------
# 現在のセットから1次関数（波長 = a * ピクセル + b）を求める
coef = np.polyfit(pxlinesD0_base, wlinesM, 1)
poly = np.poly1d(coef)


def pix2wav(x): return poly(x)


def wav2pix(w): return (w - coef[1]) / coef[0]


# ---------------------------------------------------------
# 4. プロット作成
# ---------------------------------------------------------
# インタラクティブに操作できるよう、大きめのウィンドウで表示
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
plt.subplots_adjust(hspace=0.4)

# --- 上段: 太陽スペクトル (横軸: 波長 nm) ---
ax1.plot(xm, ym, color='tab:blue', label='Solar Reference (Smoothed)')
for i, w in enumerate(wlinesM):
    ax1.axvline(w, color='tab:red', linestyle='--', alpha=0.8)
    ax1.text(w, ax1.get_ylim()[0], f' L{i}\n {w:.2f}nm', color='tab:red',
             va='bottom', ha='right', fontsize=9, fontweight='bold')

ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Normalized Intensity')
ax1.set_title('Solar Reference Spectrum')
ax1.grid(True, linestyle=':', alpha=0.7)

# --- 下段: 観測データ (横軸: ピクセル) ---
ax2.plot(x_obs, y_obs, color='black', label=f'Observed Data (Fiber {target_fiber})')
for i, p in enumerate(pxlinesD0_base):
    ax2.axvline(p, color='tab:red', linestyle='--', alpha=0.8)
    ax2.text(p, ax2.get_ylim()[0], f' L{i}\n {p}px', color='tab:red',
             va='bottom', ha='right', fontsize=9, fontweight='bold')

ax2.set_xlabel('Pixel')
ax2.set_ylabel('Intensity (Counts)')
ax2.set_title(f'Observed Spectrum (Fiber {target_fiber})')
ax2.grid(True, linestyle=':', alpha=0.7)

# 下段グラフの上部に「近似波長」の第2X軸を追加
secax = ax2.secondary_xaxis('top', functions=(pix2wav, wav2pix))
secax.set_xlabel('Approximate Wavelength (nm)')

# ★ 上段(波長)と下段(ピクセル)の表示範囲を同期させて、縦に比較しやすくする
ax2_xlim = ax2.get_xlim()
ax1.set_xlim(pix2wav(ax2_xlim[0]), pix2wav(ax2_xlim[1]))


# --- 5. クリックイベント処理（候補探し用） ---
def onclick(event):
    if event.xdata is None: return

    if event.inaxes == ax1:
        print(f"[Solar] Wavelength: {event.xdata:.3f} nm")
    elif event.inaxes == ax2:
        print(f"[Obs] Pixel: {event.xdata:.1f} px  (Approx Wavelength: {pix2wav(event.xdata):.3f} nm)")


fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()