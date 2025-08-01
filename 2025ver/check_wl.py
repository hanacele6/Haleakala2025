from astropy.io import fits
from pathlib import Path

# --- 設定 ---
# 2番目のコードが出力した.wc.fitsファイルのパスを指定
wc_file_path = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20250701/MERCURY1_tr.wc.fits")

# --- ヘッダーから波長情報を読み込んで計算 ---
if wc_file_path.exists():
    with fits.open(wc_file_path) as hdul:
        header = hdul[0].header
        try:
            naxis1 = header['NAXIS1']  # X軸のピクセル数
            crval1 = header['CRVAL1']  # 1番目のピクセルの波長
            cdelt1 = header['CDELT1']  # ピクセルごとの波長ステップ
            crpix1 = header.get('CRPIX1', 1.0) # 参照ピクセル（通常は1）

            # 波長軸を再構築
            start_wl = crval1 - (crpix1 - 1) * cdelt1
            end_wl = start_wl + (naxis1 - 1) * cdelt1

            print(f"--- 2番目のコードの出力チェック ---")
            print(f"ファイル: {wc_file_path.name}")
            print(f"波長範囲: {start_wl:.4f} nm から {end_wl:.4f} nm まで")
            print(f"------------------------------------")

        except KeyError as e:
            print(f"エラー: FITSヘッダーにWCSキーワード ({e}) が見つかりません。")
else:
    print(f"エラー: ファイルが見つかりません: {wc_file_path}")