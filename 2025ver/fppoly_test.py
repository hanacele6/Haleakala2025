import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path

from mkTotAtom import file_path

# ===================================================================
# ▼▼▼ 設定項目 (ここを書き換えてください) ▼▼▼
# ===================================================================

file_path = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20250501/")

# 1. 表示したいFITS画像のパス (元の観測データ)
#    どのファイルで問題が起きたか特定し、そのファイル名を指定してください。
fits_filename = file_path / " led01r_clf590n_ga7000fsp220_1_nhp_py.fits"

# 2. トレース情報のFITSファイルのパス
trace_filename = file_path / " led01r_clf590n_ga7000fsp220_1.fppoly.fits"
#trace_filename = file_path / "master_led_trace.fppoly_test.fits"
#trace_filename = file_path / "led01r_sp.fppoly_test.fits"
# 3. 問題が起きているファイバーの番号 (整数)
FIBER_TO_INSPECT = 103

# 4. 周辺のファイバーも表示するか (True/False)
PLOT_NEIGHBORS = True

# 5. プロットのX軸（波長方向）の表示範囲 (Noneにすると全体表示)
#    詳細に見たい場合は [1000, 1200] のように範囲を指定
X_RANGE = None

# 6. データとトレースファイルがあるディレクトリのパス
#    元のコードに合わせてあります
date = '20250501'
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
data_dir = base_dir / f"output/{date}"

# ===================================================================
# ▲▲▲ 設定はここまで ▲▲▲
# ===================================================================


# --- データの読み込み ---
try:
    # FITS画像の読み込み
    with fits.open(data_dir / fits_filename) as hdul:
        image_data = hdul[0].data
    print(f"画像ファイルを読み込みました: {fits_filename}")

    # トレース情報の読み込み
    with fits.open(data_dir / trace_filename) as hdul:
        fibp = hdul[0].data
    print(f"トレースファイルを読み込みました: {trace_filename}")

except FileNotFoundError as e:
    print(f"エラー: ファイルが見つかりません。パスが正しいか確認してください。")
    print(e)
    exit()


# --- プロット処理 ---
fig, ax = plt.subplots(figsize=(16, 9))

# FITS画像を表示
# 表示レンジを調整し、微弱な信号も見やすくする
vmin, vmax = np.percentile(image_data, [5, 99])
im = ax.imshow(image_data, origin='lower', cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)

# カラーバーを追加
fig.colorbar(im, ax=ax, label='Counts')

# X軸のピクセル番号配列
nx = image_data.shape[1]
trace_x = np.arange(nx)

# --- トレース情報を重ねてプロット ---

# 1. 問題のファイバー（赤色で表示）
trace_y_main = fibp[FIBER_TO_INSPECT, :]
ax.plot(trace_x, trace_y_main, color='red', linewidth=1.5, label=f'Fiber {FIBER_TO_INSPECT} ')

# 2. 周辺のファイバー（水色で表示）
if PLOT_NEIGHBORS:
    for offset in [-2, -1, 1, 2]:
        neighbor_fib = FIBER_TO_INSPECT + offset
        if 0 <= neighbor_fib < len(fibp):
            trace_y_neighbor = fibp[neighbor_fib, :]
            ax.plot(trace_x, trace_y_neighbor, color='cyan', linestyle='--', linewidth=0.8, label=f'Fiber {neighbor_fib}')


# --- グラフの仕上げ ---
ax.set_title(f'Image: {fits_filename}  |  Trace Overlay for Fiber {FIBER_TO_INSPECT}', fontsize=14)
ax.set_xlabel('Dispersion Axis (Pixel)', fontsize=12)
ax.set_ylabel('Spatial Axis (Pixel)', fontsize=12)

# 表示範囲の設定
if X_RANGE:
    ax.set_xlim(X_RANGE)
    # X軸の範囲に合わせてY軸の表示範囲も自動調整
    y_center = np.median(trace_y_main[X_RANGE[0]:X_RANGE[1]])
    ax.set_ylim(y_center - 20, y_center + 20)
else:
    # 全体表示の場合は、ファイバーの中心あたりに表示を合わせる
    y_center = np.median(trace_y_main)
    ax.set_ylim(y_center - 50, y_center + 50)


ax.legend()
plt.tight_layout()
plt.show()