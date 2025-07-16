import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from astropy.io import fits
from pathlib import Path
import os


def gaussian(x, amplitude, mean, stddev):
    """ガウス関数の定義"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def create_final_trace_products(data_filepath, dark_filepath, output_dir, config):
    print(f"--- 高精度トレース処理開始 ---")
    print(f"データファイル: {data_filepath.name}")
    if dark_filepath:
        print(f"ダークファイル: {dark_filepath.name}")

    try:
        with fits.open(data_filepath) as hdul:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
        if dark_filepath and dark_filepath.exists():
            with fits.open(dark_filepath) as hdul:
                data -= hdul[0].data
        elif dark_filepath:
            print(f"警告: 指定されたダークファイルが見つかりません: {dark_filepath}")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return

    # --- トレース処理の本体 (この部分は変更なし) ---
    data = median_filter(data, size=(1, 5))
    num_spatial_pts, num_disp_pts = data.shape

    nFibX, nFibY = config['nFibX'], config['nFibY']
    iFibInact = config['iFibInact']
    yFib0, yFib1, ypixFibWid = config['yFib0'], config['yFib1'], config['ypixFibWid']
    finterval = config['trace_interval_x']
    poly_order = config['poly_fit_order']

    iFib = np.arange(nFibX * nFibY)
    iFibAct = np.setdiff1d(iFib, iFibInact)
    yfibs = np.arange(len(iFib), dtype=float) * yFib1 + yFib0
    num_fibers = len(iFib)

    AARR = np.zeros((num_disp_pts, num_fibers, 3), dtype=float)
    fppoly_data = np.full((num_fibers, num_disp_pts), np.nan, dtype=np.float64)

    xpix = np.arange(num_disp_pts)
    xpixF = np.arange(0, num_disp_pts, finterval)
    if xpixF.max() != (num_disp_pts - 1):
        xpixF = np.append(xpixF, num_disp_pts - 1)

    for i in iFibAct:
        print(f"ファイバー {i}/{num_fibers - 1} を追跡中...", end="\r")
        m2 = yfibs[i]
        for j in xpixF:
            ypix1 = (np.arange(ypixFibWid, dtype=float) - ypixFibWid / 2 + 0.5 + m2).astype(int)
            if ypix1.min() < 0 or ypix1.max() >= num_spatial_pts:
                AARR[j, i, :] = np.nan
                continue
            ydat1 = data[ypix1, j]

            #Aini = [np.max(ydat1), m2, ypixFibWid / 5.0]
            Aini = [np.max(ydat1), ypixFibWid / 2, ypixFibWid / 5.0]

            try:
                #param_bounds = ([1, m2 - ypixFibWid, 0.1], [np.inf, m2 + ypixFibWid, 3])
                param_bounds = ([1, 0, 0.1], [np.inf, ypixFibWid, 3])

                par, cov = curve_fit(gaussian, np.arange(len(ypix1)), ydat1, p0=Aini, bounds=param_bounds,
                                     max_nfev=2000)
                #m2 = par[1]
                #AARR[j, i, :] = par

                absolute_center = ypix1[0] + par[1]  # 切り出し窓の開始位置 + フィットした相対位置
                m2 = absolute_center
                AARR[j, i, :] = [par[0], absolute_center, par[2]]

            except RuntimeError:
                AARR[j, i, :] = np.nan
                continue


        valid_points = np.isfinite(AARR[xpixF, i, 1])
        if np.count_nonzero(valid_points) > poly_order:
            coef = np.polyfit(xpixF[valid_points], AARR[xpixF, i, 1][valid_points], poly_order)
            smooth_y_trace = np.polyval(coef, xpix)
            smooth_y_trace = np.clip(smooth_y_trace, 0, num_spatial_pts - 1)
            AARR[:, i, 1] = smooth_y_trace
            fppoly_data[i, :] = smooth_y_trace

    print("\n--- 全ファイバーのトレース情報から pp1.fits と fppoly.fits を生成します ---")
    pp1_data = np.zeros_like(data, dtype=np.int16)
    trace_width_radius = 1
    for i in iFibAct:
        y_trace = AARR[:, i, 1]
        for x_pos in range(num_disp_pts):
            y_pos_float = y_trace[x_pos]
            if not np.isnan(y_pos_float):
                y_pos = int(round(y_pos_float))
                y_min = max(0, y_pos - trace_width_radius)
                y_max = min(num_spatial_pts, y_pos + trace_width_radius + 1)
                pp1_data[y_min:y_max, x_pos] = i + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    file_pp1 = output_dir / data_filepath.name.replace(".fits", ".pp1.fits").replace("_nhp_py", "")
    hdu_pp1 = fits.PrimaryHDU(data=pp1_data, header=header)
    hdul_pp1 = fits.HDUList([hdu_pp1])
    hdul_pp1.writeto(file_pp1, overwrite=True)
    print(f"視覚化マップを保存しました: {file_pp1}")

    file_fppoly = output_dir / data_filepath.name.replace(".fits", ".fppoly.fits").replace("_nhp_py", "")
    hdu_fppoly = fits.PrimaryHDU(data=fppoly_data)
    hdu_fppoly.header['NAXIS1'] = num_disp_pts
    hdu_fppoly.header['NAXIS2'] = num_fibers
    hdu_fppoly.header['CTYPE1'] = 'Wavelength'
    hdu_fppoly.header['CTYPE2'] = 'Fiber_ID (0-based)'
    hdul_fppoly = fits.HDUList([hdu_fppoly])
    hdul_fppoly.writeto(file_fppoly, overwrite=True)
    print(f"高精度軌跡データを保存しました: {file_fppoly}")


if __name__ == "__main__":
    # --- 基本設定 ---
    date = '20250501'
    # 親ディレクトリ
    base_output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/")
    # 今日の日付の出力ディレクトリ
    output_dir = base_output_dir / date
    # CSVファイルのパス
    csv_file_path = Path("mcparams202505.csv")

    # --- トレース処理に関する設定 ---
    # CSVの2列目の説明がこの文字列であるファイルをトレース対象とする
    TRACE_TARGET_DESCRIPTION = 'LED'
    # 使用するダークファイルのパス (固定)
    dark_file = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/dk01h_20s.sp.fits")

    # トレースの計算パラメータ
    config = {
        'nFibX': 10, 'nFibY': 12,
        'iFibInact': [6, 49, 69, 89, 94, 109, 117],
        'yFib0': 195, 'yFib1': 5.95, 'ypixFibWid': 12.0,
        'trace_interval_x': 16,
        'poly_fit_order': 6,
        'gauss_fit_radius': 6.0,
    }

    # --- メイン処理 ---
    print(f"--- トレース処理の準備 ---")
    print(f"'{csv_file_path.name}' を読み込み、説明が '{TRACE_TARGET_DESCRIPTION}' のファイルを探します...")

    try:
        df = pd.read_csv(csv_file_path)
        # CSVの列名を取得 (1列目: fits_path, 2列目: description)
        fits_col, desc_col = df.columns[0], df.columns[1]

        # 説明が一致する行をフィルタリング
        target_rows = df[df[desc_col] == TRACE_TARGET_DESCRIPTION]

        if target_rows.empty:
            print(f"エラー: CSV内に説明が '{TRACE_TARGET_DESCRIPTION}' のファイルが見つかりません。")
        else:
            # 見つかった最初のファイルを使って処理を実行
            original_fits_path_str = target_rows.iloc[0][fits_col]
            original_fits_path = Path(original_fits_path_str)

            # ホットピクセル除去後のファイル名を組み立てる
            nhp_fits_name = original_fits_path.stem + "_nhp_py.fits"
            input_file = output_dir / nhp_fits_name

            if not input_file.exists():
                print(f"エラー: トレース対象のファイルが見つかりませんでした: {input_file}")
                print("ホットピクセル除去処理が完了しているか確認してください。")
            else:
                # トレース処理関数を呼び出す
                create_final_trace_products(input_file, dark_file, output_dir, config)

    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
    except (IndexError, KeyError):
        print("エラー: CSVファイルの形式が正しくないようです。少なくとも2列（パス、説明）が必要です。")

    print("\n--- トレース処理スクリプト完了 ---")