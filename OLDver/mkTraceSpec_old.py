import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from pathlib import Path
import glob
import os


def extract_spectra_universal(science_file_pattern, fppoly_path, dark_path, output_dir, config):
    """
    fppolyの軌跡データを元に、科学画像から各ファイバーのスペクトルを抽出する。
    新旧両方のデータ形式、ダーク減算の有無、軸長の不一致、出力形式の制御に完全に対応。
    """
    print(f"--- スペクトル抽出処理開始 ---")

    # --- 1. 共通データの読み込み ---
    try:
        with fits.open(fppoly_path) as hdul:
            fppoly_data = hdul[0].data
    except FileNotFoundError as e:
        print(f"エラー: fppolyファイルが見つかりません。 {e}")
        return

    # fppolyの軸の向きを自動で判定・修正
    if fppoly_data.shape[0] > fppoly_data.shape[1]:
        print(f"  > 読み込んだfppoly.fits ({fppoly_data.shape}) は転置されていると判断し、修正します。")
        fppoly_data = fppoly_data.T
        print(f"  > fppoly.fitsを内部的に ({fppoly_data.shape}) の形にしました。")

    science_files = sorted(glob.glob(science_file_pattern))
    if not science_files:
        print(f"エラー: 指定されたパターンの科学ファイルが見つかりません: {science_file_pattern}")
        return

    print(f"> {len(science_files)}個の科学ファイルを処理します。")

    # configからパラメータを取得
    dispersion_axis = config['dispersion_axis']
    hwid = config['extraction_hwid']
    should_subtract_dark = config.get('subtract_dark', False)
    num_fibers = fppoly_data.shape[0]

    dark_frame = None
    if should_subtract_dark:
        try:
            with fits.open(dark_path) as hdul_dark:
                dark_frame = hdul_dark[0].data
            print(f"> ダークフレーム {Path(dark_path).name} を読み込みました。")
        except FileNotFoundError:
            print(f"エラー: 指定されたダークファイルが見つかりません: {dark_path}。処理を中断します。")
            return

    # --- 2. 各科学ファイルをループ処理 ---
    for science_filepath_str in science_files:
        science_filepath = Path(science_filepath_str)
        print(f"\n> 処理中: {science_filepath.name}")

        with fits.open(science_filepath) as hdul:
            science_data = hdul[0].data.astype(np.float64)
            header = hdul[0].header

        if should_subtract_dark:
            if science_data.shape != dark_frame.shape:
                print(f"  > エラー: 科学画像({science_data.shape})とダーク({dark_frame.shape})のサイズが異なります。")
                continue
            science_data -= dark_frame
            print("  > ダーク減算を実行しました。")
        else:
            print("  > ダーク減算はスキップされました。")

        # 軸長の不一致に対応
        if dispersion_axis == 0:
            num_disp_pts_science, num_spatial_pts = science_data.shape
        else:
            num_spatial_pts, num_disp_pts_science = science_data.shape
        num_disp_pts_fppoly = fppoly_data.shape[1]
        if num_disp_pts_fppoly != num_disp_pts_science:
            num_disp_pts = min(num_disp_pts_fppoly, num_disp_pts_science)
            print(f"  > 警告: fppolyと科学画像の分散軸長が異なります({num_disp_pts_fppoly} vs {num_disp_pts_science})。")
            print(f"  > 短い方の {num_disp_pts} ピクセル分だけ処理を行います。")
        else:
            num_disp_pts = num_disp_pts_science

        extracted_spectra = np.full((num_fibers, num_disp_pts), np.nan, dtype=np.float64)

        # --- 3. 各ファイバーのスペクトルを抽出 ---
        for ifib in range(num_fibers):
            trace_coords = fppoly_data[ifib, :num_disp_pts]
            rectified_strip = np.full((hwid * 2 + 1, num_disp_pts), np.nan)
            spatial_coords_axis = np.arange(num_spatial_pts)

            for i_disp in range(num_disp_pts):
                spatial_center = trace_coords[i_disp]
                if np.isnan(spatial_center): continue
                interp_points = np.arange(-hwid, hwid + 1) + spatial_center

                if dispersion_axis == 0:
                    data_slice_1d = science_data[i_disp, :]
                else:
                    data_slice_1d = science_data[:, i_disp]

                interp_func = interp1d(spatial_coords_axis, data_slice_1d,
                                       bounds_error=False, fill_value=np.nan)
                rectified_strip[:, i_disp] = interp_func(interp_points)

            signal = np.nansum(rectified_strip[hwid - 1: hwid + 2, :], axis=0)
            background = (rectified_strip[0, :] + rectified_strip[-1, :]) / 2.0
            final_spectrum = signal - background * 3.0
            extracted_spectra[ifib, :] = final_spectrum

        # --- 4. 結果をFITSファイルに保存 ---
        output_dir.mkdir(parents=True, exist_ok=True)

        # ★★★【改善点】configの設定に応じて、転置するかどうかを決定 ★★★
        should_transpose = config.get('output_transpose', False)

        if should_transpose:
            # 去年形式: (波長, ファイバー) -> 横軸:ファイバー, 縦軸:波長
            output_data = extracted_spectra.T
            header['NAXIS1'] = num_fibers
            header['NAXIS2'] = num_disp_pts
            header['CTYPE1'] = 'Fiber_ID'
            header['CTYPE2'] = 'Wavelength'
        else:
            # 今年形式: (ファイバー, 波長) -> 横軸:波長, 縦軸:ファイバー
            output_data = extracted_spectra
            header['NAXIS1'] = num_disp_pts
            header['NAXIS2'] = num_fibers
            header['CTYPE1'] = 'Wavelength'
            header['CTYPE2'] = 'Fiber_ID'

        output_filepath = output_dir / f"{science_filepath.stem}_tr.fits"
        fits.writeto(output_filepath, output_data, header, overwrite=True)
        print(f"  > 抽出したスペクトルを保存しました: {output_filepath}")

    print("\n--- 全ての処理が完了しました ---")

if __name__ == '__main__':

    # ===================================================================
    # --- 去年用データの実行例 ---
    # ===================================================================
    print("--- 去年用データ(旧形式)の処理を実行します ---")

    date_old = 'test'
    base_dir_old = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    science_pattern_old = str(base_dir_old / f"output/{date_old}/mc01_*_nhp_py.fits")
    fppoly_path_old = base_dir_old / f"output/{date_old}/fppoly1.fit"
    dark_path_old = ""
    output_dir_old = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date_old}")

    config_old = {
        'dispersion_axis': 0,
        'extraction_hwid': 5,
        'subtract_dark': False,
        'output_transpose': False,
        'output_flip_lr': True,
    }

    extract_spectra_universal(science_pattern_old, fppoly_path_old, dark_path_old, output_dir_old, config_old)