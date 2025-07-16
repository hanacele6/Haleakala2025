import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from pathlib import Path

# --- 設定 ---
# 確認したいファイルとファイバーの番号を指定してください
date = "20250501"
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
output_dir = base_dir / f"output/{date}"

# フラット補正スクリプトが生成した、規格化前のマスターフラット
# もしこのファイルがなければ、フラット補正スクリプトのデバッグ行を有効にして再度実行してください
master_flat_before_norm_path = output_dir / "debug_master_flat_BEFORE_NORM.fit"

# プロットしたいファイバーの番号 (0から数えます)
fiber_to_plot = 56

# --- プロット処理 ---
try:
    with fits.open(master_flat_before_norm_path) as hdul:
        master_flat_data = hdul[0].data

    # 指定したファイバーのスペクトルデータを取得
    spectrum = master_flat_data[fiber_to_plot, :]

    # グラフを作成
    plt.figure(figsize=(12, 6))
    plt.plot(spectrum)
    plt.title(f'Spectrum of Fiber #{fiber_to_plot} from {master_flat_before_norm_path.name}')
    plt.xlabel('Wavelength Pixel')
    plt.ylabel('Counts')
    plt.grid(True)

    # グラフを画像ファイルとして保存
    plot_filename = output_dir / f'debug_spectrum_fiber_{fiber_to_plot}.png'
    plt.savefig(plot_filename)

    print(f"スペクトルのグラフを保存しました: {plot_filename}")
    print(f"ファイバー #{fiber_to_plot} の最大値: {np.max(spectrum)}")
    print(f"ファイバー #{fiber_to_plot} の中央値: {np.median(spectrum)}")

except FileNotFoundError:
    print(f"エラー: {master_flat_before_norm_path} が見つかりません。")
    print("フラット補正スクリプトにデバッグ用の fits.writeto() が追加されているか確認してください。")