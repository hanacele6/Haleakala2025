import numpy as np
from astropy.io import fits
import sys

# ★★★ 確認したいFITSファイルのフルパスをここに指定 ★★★
filepath = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20250501/MERCURY1_f.wc.fits"

try:
    with fits.open(filepath) as hdul:
        data = hdul[0].data

        print("\n" + "=" * 50)
        print(f"ファイル {filepath} の中身を確認します。")
        print("=" * 50)
        print(f"  データ形状 (Shape): {data.shape}")
        print(f"  データ全体の平均値: {np.nanmean(data)}")
        print(f"  データ全体の最大値: {np.nanmax(data)}")
        print(f"  データ全体の最小値: {np.nanmin(data)}")
        print("=" * 50)

except FileNotFoundError:
    print(f"\nエラー: ファイルが見つかりません: {filepath}")
    sys.exit()
except Exception as e:
    print(f"\nエラー: {e}")