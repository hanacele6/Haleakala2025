import numpy as np
import pandas as pd
from astropy.io import fits
# interp1d と curve_fit は不要になります
# from scipy.interpolate import interp1d
# from scipy.optimize import curve_fit
from pathlib import Path
import os
import time
import re

# --- 設定値 (Setting Values) ---
date = '20250501'
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
output_dir = base_dir / f"output/{date}"
csv_file_path = Path("mcparams202505.csv")  # 基準となるCSVファイル

# ディレクトリが存在しない場合に作成
output_dir.mkdir(parents=True, exist_ok=True)

# --- CSVファイルとfppolyファイルの読み込み ---
try:
    df = pd.read_csv(csv_file_path)
    fits_col, desc_col = df.columns[0], df.columns[1]

    # トレース情報の基準となるfppolyファイルを特定
    led_rows = df[df[desc_col] == 'LED']
    if led_rows.empty:
        raise FileNotFoundError("CSVに 'LED' のエントリが見つかりません。fppolyファイルを特定できません。")

    led_path_str = led_rows.iloc[0][fits_col]
    fppoly_filename = Path(led_path_str).stem + ".fppoly.fits"
    fppoly_path = output_dir / fppoly_filename

    with fits.open(fppoly_path) as hdul:
        # fibp は (ファイバー数, 波長ピクセル数) の形状と仮定
        fibp = hdul[0].data
    print(f"トレース情報を読み込みました: {fppoly_path.name}")

except (FileNotFoundError, IndexError, KeyError) as e:
    print(f"エラー: 必要なファイルやCSVの列が見つかりません: {e}")
    exit()

# 処理対象のファイルリストをCSVから取得
file_list_original = df[fits_col].tolist()
ifilem = len(file_list_original)

if not file_list_original:
    print("CSVのファイルリストが空です。処理を終了します。")
    exit()

print(f"'{csv_file_path.name}' から {ifilem} 個のファイルを処理対象とします。")

# --- 定数設定 (Constants Definition) ---
NX = 2048
NY = 1024
ifibm = fibp.shape[0]

# <<< 変更点: 谷引き法で使う定数を追加 >>>
fibwid = 7  # ファイバーのフラックスを足し合わせる幅 (ピクセル)
dark_fibwid = 5  # バックグラウンド(谷)のフラックスを計算する幅 (ピクセル)

# <<< 変更点: fppoly.fitsから谷(valley)の位置を計算 >>>
# 隣り合うファイバーの中心位置の中点を谷とする
valleys = (fibp[:-1, :] + fibp[1:, :]) / 2.0
print(f"ファイバー中心位置から {valleys.shape[0]} 個の谷の位置を計算しました。")

# --- poserr.txt は不要 ---
total_start_time = time.time()

# --- ファイルごとの処理ループ ---
for ifile, original_filepath_str in enumerate(file_list_original):

    original_path = Path(original_filepath_str)
    processed_filename = original_path.stem + "_nhp_py.fits"
    data_file_path = output_dir / processed_filename

    file_start_time = time.time()
    print(f"\nファイル {ifile + 1}/{ifilem} を処理中: {processed_filename}")

    if not data_file_path.exists():
        print(f"警告: データファイル {data_file_path} が見つかりませんでした。スキップします。")
        continue

    with fits.open(data_file_path) as hdul:
        b = hdul[0].data.astype(np.float64)

    fiblall2 = np.zeros((ifibm, NX), dtype=np.float64)

    # <<< 変更点: ここから下のループ内全体を「谷引き」のロジックに置き換え >>>

    # --- ファイバーごとの処理ループ (谷引き法) ---
    for ifib in range(ifibm):
        # 各波長(ix)ごとに処理
        for ix in range(NX):
            # <<< 修正: 計算前にNaNでないかチェックする処理を追加 >>>

            center_y = fibp[ifib, ix]
            # 1. 中心のy座標自体がNaNかチェック
            if np.isnan(center_y):
                fiblall2[ifib, ix] = 0.0
                continue  # このピクセルの処理を中断し、次のixへ

            # 2. ファイバー本体のフラックスを抽出 (spDat)
            y_start = int(center_y - (fibwid - 1) / 2 + 0.5)
            y_indices = np.arange(y_start, y_start + fibwid)
            y_indices = np.clip(y_indices, 0, NY - 1)
            spDat = np.sum(b[y_indices, ix])

            total_background = 0.0  # 背景光を初期化

            # 3. バックグラウンド(谷)のフラックスを抽出 (spDk)
            try:
                # 中間のファイバーの場合
                if 0 < ifib < ifibm - 1:
                    valley1_y = valleys[ifib - 1, ix]
                    valley2_y = valleys[ifib, ix]
                    # 谷の位置がNaNでないかチェック
                    if np.isnan(valley1_y) or np.isnan(valley2_y):
                        raise ValueError("Valley is NaN")  # NaNなら例外を発生させてcatchへ

                    bg1 = np.mean(b[np.clip(np.arange(int(valley1_y - (dark_fibwid - 1) / 2 + 0.5),
                                                      int(valley1_y + (dark_fibwid + 1) / 2 + 0.5)), 0, NY - 1), ix])
                    bg2 = np.mean(b[np.clip(np.arange(int(valley2_y - (dark_fibwid - 1) / 2 + 0.5),
                                                      int(valley2_y + (dark_fibwid + 1) / 2 + 0.5)), 0, NY - 1), ix])
                    spDk_per_pixel = (bg1 + bg2) / 2.0

                # 最初のファイバーの場合
                elif ifib == 0:
                    valley_y = valleys[ifib, ix]
                    if np.isnan(valley_y):
                        raise ValueError("Valley is NaN")
                    spDk_per_pixel = np.mean(b[np.clip(np.arange(int(valley_y - (dark_fibwid - 1) / 2 + 0.5),
                                                                 int(valley_y + (dark_fibwid + 1) / 2 + 0.5)), 0,
                                                       NY - 1), ix])

                # 最後のファイバーの場合
                else:  # ifib == ifibm - 1
                    valley_y = valleys[ifib - 1, ix]
                    if np.isnan(valley_y):
                        raise ValueError("Valley is NaN")
                    spDk_per_pixel = np.mean(b[np.clip(np.arange(int(valley_y - (dark_fibwid - 1) / 2 + 0.5),
                                                                 int(valley_y + (dark_fibwid + 1) / 2 + 0.5)), 0,
                                                       NY - 1), ix])

                if not np.isnan(spDk_per_pixel):
                    total_background = spDk_per_pixel * fibwid

            except (ValueError, IndexError):
                # 谷がNaNだった場合や、その他の計算エラーが出た場合は背景を0とする
                total_background = 0.0

            # 4. 背景光を差し引く
            fiblall2[ifib, ix] = spDat - total_background



    # デバッグコードは不要になるので削除

    # CSVの2列目から説明を取得
    description = df.iloc[ifile][desc_col]
    stem = original_path.stem
    match = re.search(r'_(\d+)$', stem)
    file_num = match.group(1) if match else str(ifile + 1)

    # 新しいファイル名を組み立てる (元の結果と区別するため)
    output_file_name = f"{description}{file_num}_tr2.fits"
    output_path = output_dir / output_file_name
    hdu = fits.PrimaryHDU(fiblall2)
    hdu.writeto(output_path, overwrite=True)

    file_end_time = time.time()
    print(
        f"ファイル {ifile + 1}/{ifilem} 処理完了。 '{output_file_name}' として保存。所要時間: {file_end_time - file_start_time:.2f}秒")

total_end_time = time.time()
print(f"\nすべての処理が完了しました。全体所要時間: {total_end_time - total_start_time:.2f}秒")