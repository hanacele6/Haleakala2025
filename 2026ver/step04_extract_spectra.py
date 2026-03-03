import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
import time

def gaussian(x, amplitude, mean, stddev, offset):
    """ガウス関数の定義"""
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + offset


def run(run_info, config):
    output_dir = run_info["output_dir"]
    csv_file_path = run_info["csv_path"]

    ext_conf = config.get("extraction", {})
    hwid = ext_conf.get("hwid", 3)
    debug_threshold = ext_conf.get("debug_threshold", 0)
    max_fit_fev = ext_conf.get("max_fit_fev", 10000)
    force_rerun = config.get("pipeline", {}).get("force_rerun_extract", False)

    print(f"\n--- スペクトル抽出処理を開始します ---")

    try:
        df = pd.read_csv(csv_file_path)
        fits_col, desc_col = df.columns[0], df.columns[1]

        trace_descs = config.get("tracing", {}).get("target_descriptions", ['LED', 'HLG'])
        trace_ref_rows = df[df[desc_col].isin(trace_descs)]

        if trace_ref_rows.empty:
            raise FileNotFoundError(f"CSVに {trace_descs} が見つからず、fppolyファイルを特定できません。")

        ref_path_str = trace_ref_rows.iloc[0][fits_col]
        clean_stem = Path(ref_path_str).stem.replace("_nhp_py", "")
        fppoly_filename = f"{clean_stem}.fppoly.fits"

        # ★ 修正: トレースファイルが 1_fits/ にあるか探す
        fppoly_path = output_dir / fppoly_filename
        if not fppoly_path.exists():
            fppoly_path = output_dir / "1_fits" / fppoly_filename

        with fits.open(fppoly_path) as hdul:
            fibp = hdul[0].data
        print(f"トレース情報を読み込みました: {fppoly_filename}")

    except Exception as e:
        print(f"エラー: 抽出に必要なファイルや情報が見つかりません: {e}")
        return

    file_list_original = df[fits_col].tolist()
    ifilem = len(file_list_original)

    if not file_list_original:
        print("CSVのファイルリストが空です。")
        return

    print(f"'{csv_file_path.name}' から {ifilem} 個のファイルを処理します (hwid={hwid})。")

    NX, NY = 2048, 1024
    ifibm = fibp.shape[0]

    poserr_path = output_dir / 'poserr_python.txt'
    total_start_time = time.time()

    with open(poserr_path, 'w') as lunw2:
        for ifile, original_filepath_str in enumerate(file_list_original):
            original_path = Path(original_filepath_str)
            description = df.iloc[ifile][desc_col]

            # ★ 修正: 入力ファイルが 1_fits/ にあるか探す
            processed_filename = original_path.stem + "_nhp_py.fits"
            data_file_path = output_dir / processed_filename
            if not data_file_path.exists():
                data_file_path = output_dir / "1_fits" / processed_filename

            # ★ 修正: 出力ファイルがすでに 1_fits/ にあるかチェック
            current_count = sum(df.iloc[:ifile + 1][desc_col] == description)
            output_file_name = f"{description}{current_count}_tr.fits"
            output_path_direct = output_dir / output_file_name
            output_path_organized = output_dir / "1_fits" / output_file_name

            is_processed = output_path_direct.exists() or output_path_organized.exists()

            if is_processed and not force_rerun:
                print(f"[{ifile + 1}/{ifilem}] 処理済みスキップ: {output_file_name}")
                continue

            print(f"\n[{ifile + 1}/{ifilem}] 抽出中: {processed_filename} -> {output_file_name}")
            file_start_time = time.time()

            if not data_file_path.exists():
                print(f"  > 警告: 前処理済みファイルが見つかりません: {processed_filename}")
                continue

            # (以下、抽出のコア処理は変更なし)
            with fits.open(data_file_path) as hdul:
                b = hdul[0].data.astype(np.float64)

            fiblall2 = np.zeros((ifibm, NX), dtype=np.float64)

            for ifib in range(ifibm):
                fibpix = fibp[ifib, :]
                if np.isnan(fibpix).any(): continue

                fibl = np.zeros((hwid * 2 + 1, NX), dtype=np.float64)
                y_pixel_indices = np.arange(NY)

                for ix in range(NX):
                    spatial_data_slice = b[:, ix]
                    interp_func = interp1d(y_pixel_indices, spatial_data_slice, kind='linear', bounds_error=False,
                                           fill_value='extrapolate')
                    pos_to_interp = fibpix[ix] + np.arange(-hwid, hwid + 1)
                    fibl[:, ix] = interp_func(pos_to_interp)

                fibl2 = np.sum(fibl, axis=1)
                x_fit = np.arange(hwid * 2 + 1)
                initial_center_guess = float(np.argmax(fibl2))
                initial_guess = [np.max(fibl2) - np.min(fibl2), initial_center_guess, 1.0, np.min(fibl2)]
                lower_bounds, upper_bounds = [0, 0, 0.1, -np.inf], [np.inf, hwid * 2, hwid, np.inf]

                try:
                    params, _ = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess, bounds=(lower_bounds, upper_bounds),
                                          maxfev=max_fit_fev)
                    arr = params
                except RuntimeError:
                    arr = initial_guess

                if abs(arr[1] - hwid) > 1.5:
                    lunw2.write(f"{ifile} {ifib} {arr[1]:.6f}\n")
                    arr[1] = float(hwid)

                fibl3 = np.zeros_like(fibl)
                x_interp_data = np.arange(fibl.shape[0])
                for ix in range(NX):
                    interp_func_fibl3 = interp1d(x_interp_data, fibl[:, ix], kind='linear', bounds_error=False,
                                                 fill_value='extrapolate')
                    fibl3[:, ix] = interp_func_fibl3(arr[1] + np.arange(-hwid, hwid + 1))

                fibl4 = np.sum(fibl3[hwid - 1:hwid + 2, :], axis=0) - (fibl3[0, :] + fibl3[hwid * 2, :]) / 2.0 * 3.0
                fiblall2[ifib, :] = fibl4

            # 直下に保存（main.py の最後で自動整理されるため）
            hdu = fits.PrimaryHDU(fiblall2)
            hdu.writeto(output_path_direct, overwrite=True)
            print(f"  > 完了。所要時間: {time.time() - file_start_time:.2f}秒")

    print(f"--- 抽出処理完了 (全体所要時間: {time.time() - total_start_time:.2f}秒) ---")


if __name__ == '__main__':
    print("This script is a module.")