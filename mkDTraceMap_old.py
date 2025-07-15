import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
from pathlib import Path


# ... (gaussian関数の定義は変更なし) ...
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def refine_traces_for_old_data(raw_data_path, trace_map_path, output_dir, config):
    print(f"--- トレース精密化処理開始 ---")
    print(f"  元データ: {raw_data_path.name}")
    print(f"  トレースマップ: {trace_map_path.name}")

    # --- 1. ファイル読み込みとパラメータ設定 ---
    try:
        with fits.open(raw_data_path) as hdul:
            raw_data = hdul[0].data.astype(np.float64)
        with fits.open(trace_map_path) as hdul:
            trace_map = hdul[0].data
            header = hdul[0].header
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return

    gauss_width = config['gauss_fit_radius']
    poly_order = config['poly_fit_order']
    background_level = config['background_level_for_fit']
    correction_point = config['manual_correction_point']

    ny, nx = raw_data.shape
    iys2, iye2 = config['analysis_range_y']
    iym2 = iye2 - iys2 + 1

    # ★★★【最重要修正点】★★★
    # .max()で最大値を探すのをやめ、configから正しいファイバー数を読み込む
    num_fibers = config['number_of_fibers']

    print(f"  > {num_fibers}本のファイバーを対象に、Y={iys2}から{iye2}の範囲で処理します。")

    # ... (以降のコードは前回から変更ありません) ...
    measured_centers = np.full((num_fibers, iym2), np.nan, dtype=np.float64)
    corrected_centers = np.full((num_fibers, iym2), np.nan, dtype=np.float64)
    final_poly_traces = np.full((num_fibers, iym2), np.nan, dtype=np.float64)
    all_deviations = np.full(num_fibers, np.nan, dtype=np.float64)

    for ifib in range(1, num_fibers + 1):
        print(f"  > ファイバー {ifib}/{num_fibers} を処理中...")
        refined_y_centers = np.full(iym2, np.nan, dtype=np.float64)

        for iy in range(iys2, iye2 + 1):
            trace_row = trace_map[iy, :]
            found_x = np.where(trace_row == ifib)[0]
            if len(found_x) == 0: continue
            ifibpix = found_x[0]

            x_start = ifibpix - gauss_width
            x_end = ifibpix + gauss_width + 1
            if x_start < 0 or x_end > nx: continue

            x_local = np.arange(x_end - x_start)
            y_data = raw_data[iy, x_start:x_end]

            try:
                p0 = [y_data.max() - background_level, gauss_width, 1.0]
                popt, pcov = curve_fit(gaussian, x_local, y_data - background_level, p0=p0)
                refined_y_centers[iy - iys2] = popt[1] + x_start
            except RuntimeError:
                continue

        measured_centers[ifib - 1, :] = refined_y_centers
        corrected_trace = np.copy(refined_y_centers)

        if correction_point is not None:
            c_idx = correction_point - iys2
            if 1 < c_idx < iym2 - 2:
                neighbor_values = [
                    corrected_trace[c_idx - 2], corrected_trace[c_idx - 1],
                    corrected_trace[c_idx + 1], corrected_trace[c_idx + 2]
                ]
                if not np.all(np.isnan(neighbor_values)):
                    avg_val = np.nanmean(neighbor_values)
                    corrected_trace[c_idx] = avg_val

        valid_indices = ~np.isnan(corrected_trace)
        if np.count_nonzero(valid_indices) > poly_order:
            coeffs = np.polyfit(np.arange(iym2)[valid_indices], corrected_trace[valid_indices], poly_order)
            smoothed_trace = np.polyval(coeffs, np.arange(iym2))

            corrected_centers[ifib - 1, :] = corrected_trace
            final_poly_traces[ifib - 1, :] = smoothed_trace
            all_deviations[ifib - 1] = np.nanstd(corrected_trace - smoothed_trace)
        else:
            print(f"    > 警告: ファイバー {ifib} は有効な点が少なくフィットできませんでした。")

    print("\n--- 全ての処理が完了。結果をファイルに保存します ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    err_data = np.abs(corrected_centers - final_poly_traces)
    fits.writeto(output_dir / "err1.fits", err_data.T, header, overwrite=True)
    fits.writeto(output_dir / "fpgauss1.fits", corrected_centers.T, header, overwrite=True)
    fits.writeto(output_dir / "fppoly1.fits", final_poly_traces.T, header, overwrite=True)

    print(f"  > 平均誤差(stddev): {np.nanmean(all_deviations):.4f}")
    print(f"  > {output_dir} にファイルを保存しました。")

if __name__ == "__main__":
    # ===================================================================
    # --- ユーザー設定 (旧データ形式用) ---
    # ===================================================================
    # 1. ファイルパスを直接指定
    date = '20241119'
    base_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/data/{date}")
    output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date}")
    input_raw_data_path = output_dir / "flat1_py.fits"
    input_trace_map_path = output_dir / "pp1_py.fits"


    # 2. 解析パラメータを指定
    config = {
        # 処理対象のファイバーの総数をここで固定する
        'number_of_fibers': 94,
        # ガウスフィッティングに使う窓の半径 (IDLのgaussw=6)
        'gauss_fit_radius': 6,
        # 多項式フィッティングの次数 (IDLでは3次)
        'poly_fit_order': 3,
        # フィッティング時に差し引くおおよその背景レベル (IDLの aifib-2000)
        'background_level_for_fit': 2000.0,
        # 解析するY軸(波長)の範囲 [開始, 終了] (0-based)
        'analysis_range_y': [0, 1326],  # IDLの iys2=0, iye2=1326
        # 手動で補正する点のY座標 (IDLの369。不要なら None)
        'manual_correction_point': 369,
    }

    # --- 実行 ---
    refine_traces_for_old_data(input_raw_data_path, input_trace_map_path, output_dir, config)