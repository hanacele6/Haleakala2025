import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from pathlib import Path
import os
import time
import re

# --------------------------------------------------------------------------
# 設定項目 (Settings)
# --------------------------------------------------------------------------
date = '20250501'
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
csv_file_path = Path("mcparams202505.csv")
# ▼▼▼ 不良ファイバーのリストを定義 ▼▼▼
# このリストはご自身の環境に合わせて適宜修正してください。
IFIB_INACT = [6, 49, 69, 89, 94, 109, 117]
# --------------------------------------------------------------------------


# --- ディレクトリ設定 (Directory Settings) ---
output_dir = base_dir / f"output/{date}"
output_dir.mkdir(parents=True, exist_ok=True)


# --- ファイル読み込み (File Loading) ---
try:
    df = pd.read_csv(csv_file_path)
    fits_col, desc_col = df.columns[0], df.columns[1]

    led_rows = df[df[desc_col] == 'LED']
    if led_rows.empty:
        raise FileNotFoundError("CSVに 'LED' のエントリが見つかりません。fppolyファイルを特定できません。")

    led_path_str = led_rows.iloc[0][fits_col]
    fppoly_filename = Path(led_path_str).stem + ".fppoly.fits"
    fppoly_path = output_dir / fppoly_filename

    with fits.open(fppoly_path) as hdul:
        fibp = hdul[0].data
    print(f"トレース情報を読み込みました: {fppoly_path.name}")

except (FileNotFoundError, IndexError, KeyError) as e:
    print(f"エラー: 必要なファイルやCSVの列が見つかりません: {e}")
    exit()


# --- 処理対象ファイルのリストアップ (List files to process) ---
file_list_original = df[fits_col].tolist()
ifilem = len(file_list_original)
print(f"'{csv_file_path.name}' から {ifilem} 個のファイルを処理対象とします。")


# --- 定数とファイバーリストの設定 (Constants and Fiber Lists) ---
NX = fibp.shape[1]
NY = 2048  # 初期値。FITSヘッダーから読み込めれば上書きされる
ifibm = fibp.shape[0]
hwid = 2      # 主信号の積分幅の半分 (合計 2*hwid+1 ピクセル)
bg_hwid = 2   # 背景信号の積分幅の半分 (合計 2*bg_hwid+1 ピクセル)

# アクティブなファイバーのリストを作成
iFib = np.arange(ifibm)
iFibAct = np.setdiff1d(iFib, IFIB_INACT)


# --- メイン処理開始 ---
total_start_time = time.time()

# --- ファイルごとの処理ループ (Main loop for each file) ---
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
        original_header = hdul[0].header
        if 'NAXIS2' in original_header:
            NY = original_header['NAXIS2']

    fiblall2 = np.full((ifibm, NX), np.nan, dtype=np.float64) # 不良ファイバーはNaNのままにする


    # ==================================================================
    # --- Xピクセルごとのループ (Outer loop for each X-pixel) ---
    for ix in range(NX):
        # 現在のX列における、アクティブファイバーのY座標を取得
        active_fiber_y_coords = fibp[iFibAct, ix]

        # このX列に対する補間関数を作成 (一度だけ)
        interp_func = interp1d(
            iFibAct,
            active_fiber_y_coords,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

        # --- アクティブファイバーごとのループ (Inner loop for each active fiber) ---
        for ifib in iFibAct:
            # --- 1. 主信号 (spDat) の計算 ---
            center_y_main = np.round(fibp[ifib, ix]).astype(int)
            main_signal_window = np.clip(center_y_main + np.arange(-hwid, hwid + 1), 0, NY - 1)
            spDat_ix = np.sum(b[main_signal_window, ix])

            # --- 2. 背景 (spDk) の計算 ---
            # a. 補間関数を使って谷の位置を計算
            left_valley_pos = interp_func(ifib - 0.5)
            right_valley_pos = interp_func(ifib + 0.5)

            # b. 谷の周りのピクセル値から最小値を取得
            center_y_left = np.round(left_valley_pos).astype(int)
            left_valley_window = np.clip(center_y_left + np.arange(-bg_hwid, bg_hwid + 1), 0, NY - 1)
            left_min = np.min(b[left_valley_window, ix])

            center_y_right = np.round(right_valley_pos).astype(int)
            right_valley_window = np.clip(center_y_right + np.arange(-bg_hwid, bg_hwid + 1), 0, NY - 1)
            right_min = np.min(b[right_valley_window, ix])

            # c. 1ピクセルあたりの背景レベルを決定
            background_level_per_pixel = (left_min + right_min) / 2.0

            # d. 背景レベルを主信号の積分幅でスケーリング
            integration_width = hwid * 2 + 1
            spDk_ix = background_level_per_pixel * integration_width

            # --- 3. 背景を引いた最終的な値を格納 ---
            fiblall2[ifib, ix] = spDat_ix - spDk_ix

    # --- ファイル名生成と保存 (Generate Filename and Save) ---
    description = df.iloc[ifile][desc_col]
    stem = original_path.stem
    match = re.search(r'_(\d+)$', stem)
    file_num = match.group(1) if match else str(ifile + 1)
    output_file_name = f"{description}{file_num}_tr.fits"
    output_path = output_dir / output_file_name

    # 新しいデータ配列と元のヘッダーでHDUを作成
    hdu = fits.PrimaryHDU(fiblall2.astype(np.float32), header=original_header)
    hdu.header['HISTORY'] = 'Background subtracted by Python script (ver. interp).'
    hdu.writeto(output_path, overwrite=True)

    file_end_time = time.time()
    print(
        f"ファイル {ifile + 1}/{ifilem} 処理完了。 '{output_file_name}' として保存。所要時間: {file_end_time - file_start_time:.2f}秒")

total_end_time = time.time()
print(f"\nすべての処理が完了しました。全体所要時間: {total_end_time - total_start_time:.2f}秒")