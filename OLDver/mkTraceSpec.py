import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from pathlib import Path
import glob
import os


def extract_spectra_universal(science_file_pattern, fppoly_path, dark_path, output_dir, config):
    """
    fppolyの軌跡データを元に、科学画像から各ファイバーのスペクトルを抽出する。
    新旧両方のデータ形式に対応し、fppolyの出力形式も制御可能。
    """
    print(f"--- スペクトル抽出処理開始 ---")

    # --- 1. 共通データの読み込み ---
    try:
        with fits.open(fppoly_path) as hdul:
            fppoly_data = hdul[0].data
        with fits.open(dark_path) as hdul:
            dark_frame = hdul[0].data
    except FileNotFoundError as e:
        print(f"エラー: fppolyまたはダークファイルが見つかりません。 {e}")
        return

    science_files = sorted(glob.glob(science_file_pattern))
    if not science_files:
        print(f"エラー: 指定されたパターンの科学ファイルが見つかりません: {science_file_pattern}")
        return

    print(f"> {len(science_files)}個の科学ファイルを処理します。")

    # configからパラメータを取得
    dispersion_axis = config['dispersion_axis']
    hwid = config['extraction_hwid']
    num_fibers = fppoly_data.shape[0]

    # --- 2. 各科学ファイルをループ処理 ---
    for science_filepath_str in science_files:
        science_filepath = Path(science_filepath_str)
        print(f"\n> 処理中: {science_filepath.name}")

        with fits.open(science_filepath) as hdul:
            science_data = hdul[0].data.astype(np.float64) - dark_frame
            header = hdul[0].header

        if dispersion_axis == 0:
            num_disp_pts, num_spatial_pts = science_data.shape
        else:
            num_spatial_pts, num_disp_pts = science_data.shape

        if fppoly_data.shape[1] != num_disp_pts:
            print(f"警告: fppolyの分散軸長({fppoly_data.shape[1]})がデータ({num_disp_pts})と異なります。")
            continue

        extracted_spectra = np.full((num_fibers, num_disp_pts), np.nan, dtype=np.float64)

        # --- 3. 各ファイバーのスペクトルを抽出 ---
        for ifib in range(num_fibers):
            trace_coords = fppoly_data[ifib, :]
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

            # --- 4. アパーチャ抽出と背景除去 ---
            signal = np.nansum(rectified_strip[hwid - 1: hwid + 2, :], axis=0)
            background = (rectified_strip[0, :] + rectified_strip[-1, :]) / 2.0
            final_spectrum = signal - background * 3.0
            extracted_spectra[ifib, :] = final_spectrum

        # --- 5. 結果をFITSファイルに保存 ---
        output_dir.mkdir(parents=True, exist_ok=True)

        # ★★★【改善点】configの設定に応じて、転置するかどうかを決定 ★★★
        should_transpose = config.get('output_fppoly_transpose', False)

        if should_transpose:
            # 去年形式: (波長, ファイバー)
            output_data = extracted_spectra.T
            header['NAXIS1'] = num_fibers
            header['NAXIS2'] = num_disp_pts
            header['CTYPE1'] = 'Fiber_ID'
            header['CTYPE2'] = 'Wavelength'
        else:
            # 今年形式: (ファイバー, 波長)
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
    # --- ★★★ ここを切り替えて実行 ★★★ ---
    # ===================================================================

    # 【今年用データ（新形式）の実行例】

    # 1. ファイルの場所とパターン
    date_new = '20250501'
    base_dir_new = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    science_pattern_new = str(base_dir_new / f"output/{date_new}/ me01r_clf590n_ga7000fsp220_*_nhp_py.fits")
    fppoly_path_new = base_dir_new / f"output/{date_new}/ led01r_clf590n_ga7000fsp220_1_nhp_py.fppoly.fits"
    dark_path_new = base_dir_new / "dk01h_20s.sp.fits"
    output_dir_new = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date_new}")

    # 2. 解析パラメータ
    config_new = {
        'dispersion_axis': 1,
        'extraction_hwid': 5,
        'output_fppoly_transpose': False,  # ★新形式は転置しない
    }

    # --- 実行 ---
    extract_spectra_universal(science_pattern_new, fppoly_path_new, dark_path_new, output_dir_new, config_new)

    # # 【去年用データ（旧形式）の実行例】
    # # こちらを実行する場合は、上のブロックをコメントアウトし、下のブロックのコメントを外す

    # # 1. ファイルの場所とパターン
    # base_dir_old = Path("./")
    # science_pattern_old = str(base_dir_old / "data_old/science_*.fits")
    # fppoly_path_old = base_dir_old / "trace2_output_old/fppoly.fit"
    # dark_path_old = base_dir_old / "darks/dark_old.fits"
    # output_dir_old = base_dir_old / "extracted_spectra_old"

    # # 2. 解析パラメータ
    # config_old = {
    #     'dispersion_axis': 0,
    #     'extraction_hwid': 5,
    #     'output_fppoly_transpose': True, # ★旧形式は転置する
    # }

    # # --- 実行 ---
    # extract_spectra_universal(science_pattern_old, fppoly_path_old, dark_path_old, output_dir_old, config_old)