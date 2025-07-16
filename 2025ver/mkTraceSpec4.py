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

# --- 定数設定 (Constants) ---
NX = fibp.shape[1]
NY = 2048  # 初期値。FITSヘッダーから読み込めれば上書きされる
ifibm = fibp.shape[0]
hwid = 2  # 主信号の積分幅の半分 (合計 2*hwid+1 ピクセル)

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

    # ▼▼▼【変更点 1/2】データと一緒にヘッダーも読み込む ▼▼▼
    with fits.open(data_file_path) as hdul:
        b = hdul[0].data.astype(np.float64)
        original_header = hdul[0].header  # 元のヘッダーを読み込む
        if 'NAXIS2' in original_header:
            NY = original_header['NAXIS2']
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    fiblall2 = np.zeros((ifibm, NX), dtype=np.float64)
    y_pixel_indices = np.arange(NY)

    # --- ファイバーごとの処理ループ (Loop for each fiber) ---
    for ifib in range(ifibm):
        current_fiber_trace = fibp[ifib, :]
        if np.isnan(current_fiber_trace).any():
            continue

        # ==================================================================
        # ★★★ mkFibSpec4b.pyのロジックを再現した背景引き ★★★
        # ==================================================================
        spDat = np.zeros(NX)  # 主信号スペクトル
        spDk = np.zeros(NX)  # 背景スペクトル (スケーリング後)

        # --- Xピクセルごとのループ ---
        for ix in range(NX):
            # 高速化のため、補間関数はXピクセルごとに生成
            #interp_func_col = interp1d(y_pixel_indices, b[:, ix],kind='linear', bounds_error=False, fill_value=0.0)

            # --- 1. 主信号 (spDat) の計算 ---
            #    トレースの中心周りを積分幅で合計する
            #main_signal_window = current_fiber_trace[ix] + np.arange(-hwid, hwid + 1)
            #spDat[ix] = np.sum(interp_func_col(main_signal_window))

            #    トレースの中心位置を最も近い整数ピクセルに変換
            center_y_main = np.round(current_fiber_trace[ix]).astype(int)
            #    中心の整数ピクセルの周りに積分ウィンドウを作成
            main_signal_window_indices = center_y_main + np.arange(-hwid, hwid + 1)
            #    画像の範囲外に出ないようにインデックスをクリップ
            main_signal_window_indices = np.clip(main_signal_window_indices, 0, NY - 1)
            #    データ配列bから直接値を合計
            spDat[ix] = np.sum(b[main_signal_window_indices, ix])

            # --- 2. 背景 (spDk) の計算 ---
            # a. 谷の位置を計算
            left_valley_pos = (current_fiber_trace[ix] + fibp[ifib - 1, ix]) / 2.0 if ifib > 0 else None
            right_valley_pos = (current_fiber_trace[ix] + fibp[ifib + 1, ix]) / 2.0 if ifib < ifibm - 1 else None

            # b. 谷の周りのピクセル値を取得し、最小値をとる
            #    mkFibSpec4b.pyに合わせて背景の積分幅を5ピクセルに固定
            bg_hwid = 2  # (合計5ピクセル)

            #left_min = np.inf
            #if left_valley_pos is not None:
            #    left_valley_window = left_valley_pos + np.arange(-bg_hwid, bg_hwid + 1)
            #    left_min = np.min(interp_func_col(left_valley_window))

            left_min = np.inf
            if left_valley_pos is not None:
                # 谷の位置を最も近い整数ピクセルに変換
                center_y_left = np.round(left_valley_pos).astype(int)
                # 整数ピクセルの周りにウィンドウを作成
                left_valley_window_indices = center_y_left + np.arange(-bg_hwid, bg_hwid + 1)
                # インデックスをクリップ
                left_valley_window_indices = np.clip(left_valley_window_indices, 0, NY - 1)
                # データ配列bから直接最小値を取得
                left_min = np.min(b[left_valley_window_indices, ix])

            #right_min = np.inf
            #if right_valley_pos is not None:
            #    right_valley_window = right_valley_pos + np.arange(-bg_hwid, bg_hwid + 1)
            #    right_min = np.min(interp_func_col(right_valley_window))

            right_min = np.inf
            if right_valley_pos is not None:
                # 谷の位置を最も近い整数ピクセルに変換
                center_y_right = np.round(right_valley_pos).astype(int)
                # 整数ピクセルの周りにウィンドウを作成
                right_valley_window_indices = center_y_right + np.arange(-bg_hwid, bg_hwid + 1)
                # インデックスをクリップ
                right_valley_window_indices = np.clip(right_valley_window_indices, 0, NY - 1)
                # データ配列bから直接最小値を取得
                right_min = np.min(b[right_valley_window_indices, ix])

            # c. 1ピクセルあたりの背景レベルを決定
            background_level_per_pixel = 0.0
            if np.isfinite(left_min) and np.isfinite(right_min):
                background_level_per_pixel = (left_min + right_min) / 2.0
            elif np.isfinite(left_min):
                background_level_per_pixel = left_min
            elif np.isfinite(right_min):
                background_level_per_pixel = right_min

            # d. 背景レベルを主信号の積分幅でスケーリング
            integration_width = hwid * 2 + 1
            spDk[ix] = background_level_per_pixel * integration_width

        # --- 3. ファイバー全体のスペクトルから背景を引く ---
        final_spectrum = spDat - spDk
        #final_spectrum[final_spectrum < 0] = 0  # 負の値を0にクリップ

        fiblall2[ifib, :] = final_spectrum

    # --- ファイル名生成と保存 (Generate Filename and Save) ---
    description = df.iloc[ifile][desc_col]
    stem = original_path.stem
    match = re.search(r'_(\d+)$', stem)
    file_num = match.group(1) if match else str(ifile + 1)
    output_file_name = f"{description}{file_num}_tr.fits"

    output_path = output_dir / output_file_name

    # ▼▼▼【変更点 2/2】保存時に読み込んだヘッダーを渡す ▼▼▼
    # 新しいデータ配列と元のヘッダーでHDUを作成する
    # AstropyがNAXISなどの構造情報は自動で更新してくれる
    hdu = fits.PrimaryHDU(fiblall2.astype(np.float32), header=original_header)

    # このスクリプトで行った処理をHISTORYとして追記
    hdu.header['HISTORY'] = 'Background subtracted by Python script.'

    hdu.writeto(output_path, overwrite=True)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    file_end_time = time.time()
    print(
        f"ファイル {ifile + 1}/{ifilem} 処理完了。 '{output_file_name}' として保存。所要時間: {file_end_time - file_start_time:.2f}秒")

total_end_time = time.time()
print(f"\nすべての処理が完了しました。全体所要時間: {total_end_time - total_start_time:.2f}秒")
