import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.io import fits
from pathlib import Path
import glob
import os


# IDLのnterms=4に相当する、ベースライン付きのガウス関数
def gaussian_with_base(x, amplitude, mean, stddev, base):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + base


def extract_spectra_final_IDL_replica(science_file_pattern, fppoly_path, dark_path, output_dir, config):
    print(f"--- スペクトル抽出処理開始 (IDL完全再現版) ---")

    # --- 1. 共通データの読み込み ---
    try:
        with fits.open(fppoly_path) as hdul:
            fppoly_data = hdul[0].data
        if fppoly_data.shape[0] > fppoly_data.shape[1]:
            fppoly_data = fppoly_data.T
    except FileNotFoundError as e:
        print(f"エラー: fppolyファイルが見つかりません。 {e}")
        return

    science_files = sorted(glob.glob(science_file_pattern))
    if not science_files:
        print(f"エラー: 指定されたパターンの科学ファイルが見つかりません: {science_file_pattern}")
        return
    print(f"> {len(science_files)}個の科学ファイルを処理します。")

    dispersion_axis = config['dispersion_axis']
    hwid = config['extraction_hwid']
    should_subtract_dark = config.get('subtract_dark', False)
    num_fibers = fppoly_data.shape[0]
    dark_frame = None
    if should_subtract_dark:
        with fits.open(dark_path) as hdul_dark:
            dark_frame = hdul_dark[0].data

    # --- 2. 各科学ファイルをループ処理 ---
    for science_filepath_str in science_files:
        science_filepath = Path(science_filepath_str)
        print(f"\n> 処理中: {science_filepath.name}")

        with fits.open(science_filepath) as hdul:
            science_data = hdul[0].data.astype(np.float64)
            header = hdul[0].header

        if should_subtract_dark:
            if science_data.shape != dark_frame.shape: continue
            science_data -= dark_frame

        if dispersion_axis == 0:
            num_disp_pts, num_spatial_pts = science_data.shape
        else:
            num_spatial_pts, num_disp_pts = science_data.shape
        if fppoly_data.shape[1] != num_disp_pts:
            num_disp_pts = min(fppoly_data.shape[1], num_disp_pts)
            print(f"  > 警告: fppolyと科学画像の分散軸長が異なります。短い方の {num_disp_pts} ピクセル分だけ処理します。")

        extracted_spectra = np.full((num_fibers, num_disp_pts), np.nan, dtype=np.float64)

        # --- 3. 各ファイバーのスペクトルを抽出 ---
        for ifib in range(num_fibers):
            print(f"\r  > ファイバー {ifib + 1}/{num_fibers} を処理中...", end="")
            trace_coords = fppoly_data[ifib, :num_disp_pts]

            # 3a. 【第1パス】仮の切り抜きデータを作成
            provisional_strip = np.full((hwid * 2 + 1, num_disp_pts), np.nan)
            spatial_coords_axis = np.arange(num_spatial_pts)
            for i_disp in range(num_disp_pts):
                spatial_center = trace_coords[i_disp]
                if np.isnan(spatial_center): continue
                interp_points = np.arange(-hwid, hwid + 1) + spatial_center
                if dispersion_axis == 0:
                    data_slice_1d = science_data[i_disp, :]
                else:
                    data_slice_1d = science_data[:, i_disp]
                # 正しい状態（fill_valueがnp.nanになっている）
                interp_func = interp1d(spatial_coords_axis, data_slice_1d, bounds_error=False, fill_value=np.nan)
                provisional_strip[:, i_disp] = interp_func(interp_points)

            # 3b. 【ズレ量の計算】IDLの品質チェックと強制補正を再現
            avg_profile = np.nanmean(provisional_strip, axis=1)
            valid_profile_points = ~np.isnan(avg_profile)

            center_in_strip = float(hwid)
            if np.count_nonzero(valid_profile_points) > 4:
                try:
                    x_profile = np.arange(hwid * 2 + 1)
                    base_guess = np.nanmin(avg_profile)
                    amp_guess = np.nanmax(avg_profile) - base_guess
                    p0 = [amp_guess, hwid, 1.0, base_guess]
                    par, cov = curve_fit(gaussian_with_base, x_profile[valid_profile_points],
                                         avg_profile[valid_profile_points], p0=p0)

                    # ★★★ IDLの異常値処理を再現 ★★★
                    if abs(par[1] - hwid) > 1.5:
                        center_in_strip = 3.0  # フィット結果を捨て、3.0に強制設定
                    else:
                        center_in_strip = par[1]
                except RuntimeError:
                    center_in_strip = float(hwid)

            # 3c. 【第2パス】IDLの「二重補間」を再現
            final_rectified_strip = np.full((hwid * 2 + 1, num_disp_pts), np.nan)
            strip_coords_axis = np.arange(hwid * 2 + 1)
            for i_disp in range(num_disp_pts):
                interp_points_relative = np.arange(-hwid, hwid + 1) + center_in_strip

                interp_points_relative = np.clip(interp_points_relative, 0, hwid * 2)

                data_slice_1d = provisional_strip[:, i_disp]
                valid_points = ~np.isnan(data_slice_1d)
                if np.count_nonzero(valid_points) > 1:
                    interp_func = interp1d(strip_coords_axis[valid_points], data_slice_1d[valid_points],
                                           bounds_error=False, fill_value=np.nan)
                    final_rectified_strip[:, i_disp] = interp_func(interp_points_relative)

            # 4. アパーチャ抽出と背景除去
            signal = np.nansum(final_rectified_strip[hwid - 1: hwid + 2, :], axis=0)
            background = (final_rectified_strip[0, :] + final_rectified_strip[-1, :]) / 2.0
            background[np.isnan(background)] = 0
            final_spectrum = signal - background * 3.0
            extracted_spectra[ifib, :] = final_spectrum

        print()  # 改行
        # --- 5. 結果をFITSファイルに保存 ---
        output_dir.mkdir(parents=True, exist_ok=True)
        should_transpose = config.get('output_transpose', False)
        should_flip_lr = config.get('output_flip_lr', False)

        if should_transpose:
            output_data = extracted_spectra.T
        else:
            output_data = extracted_spectra

        if should_flip_lr:
            print("  > 出力データの左右を反転します。")
            output_data = np.fliplr(output_data)

        header['NAXIS1'] = output_data.shape[1]
        header['NAXIS2'] = output_data.shape[0]
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
        'output_transpose': False,  # 縦軸=波長, 横軸=ファイバー
        'output_flip_lr': True, #　左右を入れ替え
    }

    extract_spectra_final_IDL_replica(science_pattern_old, fppoly_path_old, dark_path_old, output_dir_old, config_old)