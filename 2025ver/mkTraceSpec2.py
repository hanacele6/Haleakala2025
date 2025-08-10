import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
import os
import time
import re  # 番号を抽出するためにインポート

# --- 設定値 (Setting Values) ---
date = '20150223' #7mada
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
output_dir = base_dir / f"output/{date}"
csv_file_path = base_dir / "2025ver" / f"mcparams{date}.csv"  # 基準となるCSVファイル

# ディレクトリが存在しない場合に作成
output_dir.mkdir(parents=True, exist_ok=True)

# --- CSVファイルとfppolyファイルの読み込み ---
try:
    df = pd.read_csv(csv_file_path)
    fits_col, desc_col = df.columns[0], df.columns[1]

    # トレース情報の基準となるfppolyファイルを特定 (LEDファイルから名前を推測)
    led_rows = df[df[desc_col] == 'LED']
    if led_rows.empty:
        raise FileNotFoundError("CSVに 'LED' のエントリが見つかりません。fppolyファイルを特定できません。")

    # LEDの最初のファイル名からfppolyファイル名を生成
    led_path_str = led_rows.iloc[0][fits_col]
    fppoly_filename = Path(led_path_str).stem + ".fppoly.fits"
    fppoly_path = output_dir / fppoly_filename

    with fits.open(fppoly_path) as hdul:
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
hwid = 3

type_counters = {}

# --- Gauss関数定義 (Gaussian function for curve_fit) ---
def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + offset


# --- poserr.txt ファイルの書き込み準備 ---
poserr_path = output_dir / 'poserr_python.txt'
with open(poserr_path, 'w') as lunw2:
    total_start_time = time.time()

    # --- ファイルごとの処理ループ ---
    for ifile, original_filepath_str in enumerate(file_list_original):

        # ホットピクセル除去後のファイルパスを生成
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

        # --- ファイバーごとの処理ループ ---
        for ifib in range(ifibm):
            fibpix = fibp[ifib, :]

            if np.isnan(fibpix).any():
                fiblall2[ifib, :] = 0.0
                continue

            fibl = np.zeros((hwid * 2 + 1, NX), dtype=np.float64)

            y_pixel_indices = np.arange(NY)
            for ix in range(NX):
                spatial_data_slice = b[:, ix]
                interp_func = interp1d(
                    y_pixel_indices, spatial_data_slice,
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                pos_to_interp = fibpix[ix] + np.arange(-hwid, hwid + 1)
                fibl[:, ix] = interp_func(pos_to_interp)

            fibl2 = np.sum(fibl, axis=1)
            x_fit = np.arange(hwid * 2 + 1)
            #initial_guess = [np.max(fibl2) - np.min(fibl2), float(hwid), 1.0, np.min(fibl2)]

            initial_center_guess = float(np.argmax(fibl2))

            initial_guess = [
                np.max(fibl2) - np.min(fibl2),  # 振幅 (Amplitude)
                initial_center_guess,  # 中心 (Mean)
                1.0,  # 幅 (Stddev)
                np.min(fibl2)  # オフセット (Offset)
            ]

            lower_bounds = [0, 0, 0.1, -np.inf]  # 中心μは0以上
            upper_bounds = [np.inf, hwid * 2, hwid, np.inf]  # 中心μは4以下、幅σは2以下

            try:
                params, _ = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess, bounds=(lower_bounds, upper_bounds),  maxfev=10000)
                arr = params
            except RuntimeError:
                arr = initial_guess

            if abs(arr[1] - hwid) > 1.5:
                lunw2.write(f"{ifile} {ifib} {arr[1]:.6f}\n")
                arr[1] = float(hwid)

            fibl3 = np.zeros_like(fibl)
            x_interp_data = np.arange(fibl.shape[0])
            for ix in range(NX):
                interp_func_fibl3 = interp1d(
                    x_interp_data, fibl[:, ix], kind='linear',
                    bounds_error=False, fill_value='extrapolate'
                )
                fibl3[:, ix] = interp_func_fibl3(arr[1] + np.arange(-hwid, hwid + 1))

            #fibl4 = fibl3[hwid, :] - (fibl3[0, :] + fibl3[hwid * 2, :]) / 2.0
            #fibl4 = np.sum(fibl3[hwid - 1:hwid + 2, :]) - (fibl3[0, :] + fibl3[hwid * 2, :]) / 2.0 * 3.0
            fibl4 = np.sum(fibl3[hwid - 1:hwid + 2, :], axis=0) - (fibl3[0, :] + fibl3[hwid * 2, :]) / 2.0 * 3.0
            #background = np.min(fibl3, axis=0)
            # 中心の値から、その背景値を引く
            #fibl4 = fibl3[hwid, :] - background
            
            fiblall2[ifib, :] = fibl4

            # ▼▼▼【デバッグ用】ここから追加 ▼▼▼
            THRESHOLD = 0  # 情報を表示する閾値。-10000などに調整してください。

            # 計算結果(fibl4)の中に、閾値より小さい値があるかチェック
            bad_indices = np.where(fibl4 < THRESHOLD)[0]

            # 異常値が1つでも見つかった場合
            if len(bad_indices) > 0:
                # 最初の異常値について情報を表示（たくさんあっても全部表示しないように）
                ix_problem = bad_indices[0]

                print(f"\n--- 異常値検出 [ファイル:{ifile}, ファイバー:{ifib}] ---")
                print(f"  波長ピクセル(ix) = {ix_problem} で 値 = {fibl4[ix_problem]:.1f} を検出しました。")
                print(f"  このファイバーのガウスフィット結果 (振幅A, 中心μ, 幅σ, オフセットC) は以下の通りです:")
                print(f"  A={arr[0]:.2f}, μ={arr[1]:.2f}, σ={arr[2]:.2f}, C={arr[3]:.2f}")

                # さらに詳細な情報として、その波長における5ピクセルの値を表示
                print(f"\n  問題の波長(ix={ix_problem})での5ピクセルの値(fibl3)は以下の通りです:")
                print(
                    f"  [ 上から順: {fibl3[0, ix_problem]:.1f}, {fibl3[1, ix_problem]:.1f}, {fibl3[2, ix_problem]:.1f}, {fibl3[3, ix_problem]:.1f}, {fibl3[4, ix_problem]:.1f} ]")
                print(f"--------------------------------------------------")
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲



        # CSVの2列目から説明を取得
        description = df.iloc[ifile][desc_col]

        # 説明(description)ごとのカウンターを更新し、番号を取得
        # .get(description, 0) で、初めてのキーなら0を取得
        current_count = type_counters.get(description, 0) + 1
        type_counters[description] = current_count  # カウンターを更新
        file_num = str(current_count)

        # 新しいファイル名を組み立てる
        output_file_name = f"{description}{file_num}_tr.fits"


        output_path = output_dir / output_file_name
        hdu = fits.PrimaryHDU(fiblall2)
        hdu.writeto(output_path, overwrite=True)

        file_end_time = time.time()
        print(
            f"ファイル {ifile + 1}/{ifilem} 処理完了。 '{output_file_name}' として保存。所要時間: {file_end_time - file_start_time:.2f}秒")

    total_end_time = time.time()
    print(f"\nすべての処理が完了しました。全体所要時間: {total_end_time - total_start_time:.2f}秒")