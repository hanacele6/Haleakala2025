import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from astropy.io import fits
from pathlib import Path
import os


def gaussian_with_base(x, amplitude, mean, stddev, base):
    """ベースライン成分(base)を含むガウス関数の定義"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + base


def create_final_trace_products(data_filepath, dark_filepath, output_dir, config):
    """
    背景変動に強く、軌跡の平滑化が安定した、高機能なトレース処理を行う。
    視覚的なpp1.fitsと、科学的なfppoly.fitsの両方を生成する。
    """
    print(f"--- 高精度トレース処理開始 (最終版) ---")
    print(f"データファイル: {data_filepath.name}")
    print(f"ダークファイル: {dark_filepath.name}")

    # --- 1. 準備：ファイル読み込みとパラメータ設定 ---
    try:
        with fits.open(data_filepath) as hdul:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
        with fits.open(dark_filepath) as hdul:
            data -= hdul[0].data
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return

    data = median_filter(data, size=(1, 5))
    num_spatial_pts, num_disp_pts = data.shape

    # configからパラメータを読み込む
    nFibX, nFibY, iFibInact = config['nFibX'], config['nFibY'], config['iFibInact']
    yFib0, yFib1, ypixFibWid = config['yFib0'], config['yFib1'], config['ypixFibWid']
    finterval = config['trace_interval_x']
    gauss_width = config['gauss_fit_radius']
    poly_order = config['poly_fit_order']
    ref_start, ref_end = config['reference_brightness_range']
    cutoff_fraction = config['trace_cutoff_fraction']

    # ファイバーの基本情報を設定
    iFib = np.arange(nFibX * nFibY)
    iFibAct = np.setdiff1d(iFib, iFibInact)
    yfibs = np.arange(len(iFib), dtype=float) * yFib1 + yFib0
    num_fibers = len(iFib)

    # 疎な点(xpixF)での結果を格納する配列。ガウス関数の4パラメータを格納
    xpix = np.arange(num_disp_pts)
    xpixF = np.arange(0, num_disp_pts, finterval)
    if xpixF.max() != (num_disp_pts - 1):
        xpixF = np.append(xpixF, num_disp_pts - 1)
    AARR_sparse = np.full((len(xpixF), num_fibers, 4), np.nan, dtype=np.float64)

    # --- 2. 各ファイバーを疎な点で追跡 ---
    for i in iFibAct:
        print(f"ファイバー {i}/{num_fibers - 1} の一次追跡...", end="")
        m2 = yfibs[i]  # 初期位置の推測値
        for j_idx, j_abs in enumerate(xpixF):
            ypix1 = (np.arange(ypixFibWid, dtype=float) - ypixFibWid / 2 + 0.5 + m2).astype(int)
            if ypix1.min() < 0 or ypix1.max() >= num_spatial_pts:
                continue

            ydat1 = data[ypix1, j_abs]

            # ★★★ ここからが修正部分 ★★★
            # 1. ベースラインの初期値を安全な範囲に収める
            base_guess = np.min(ydat1)
            if base_guess < 0: base_guess = 0  # 負なら0にする

            # 2. 振幅の初期値を安全な範囲に収める
            amp_guess = np.max(ydat1) - base_guess  # 修正後のベースラインを使う
            if amp_guess < 1: amp_guess = 1  # 1未満なら1にする

            # 安全な値を使って初期推測値リストを作成
            Aini = [amp_guess, m2, ypixFibWid / 5.0, base_guess]
            # ★★★ ここまでが修正部分 ★★★
            try:
                param_bounds = ([1, m2 - ypixFibWid, 0.1, 0], [1e5, m2 + ypixFibWid, 3, 1e5])
                par, cov = curve_fit(gaussian_with_base, ypix1, ydat1, p0=Aini, bounds=param_bounds)
                m2 = par[1]  # 中心位置を更新
                AARR_sparse[j_idx, i, :] = par
            except RuntimeError:
                continue
        print(" 完了")

    # --- 3. 相対的な閾値でトレースをフィルタリング ---
    print("\n--- 各ファイバーのトレースを相対的閾値でフィルタリングします ---")
    ref_indices = np.where((xpixF >= ref_start) & (xpixF <= ref_end))[0]
    for i in iFibAct:
        amplitudes = AARR_sparse[:, i, 0]  # 振幅は0番目のパラメータ
        ref_amps = amplitudes[ref_indices]
        valid_ref_amps = ref_amps[np.isfinite(ref_amps)]
        if len(valid_ref_amps) < 3:
            print(f"  > 警告: ファイバー {i} は参照範囲内の信号が弱く、フィルタリングをスキップします。")
            continue

        reference_brightness = np.median(valid_ref_amps)
        cutoff_threshold = reference_brightness * cutoff_fraction
        below_threshold_indices = np.where(amplitudes < cutoff_threshold)[0]

        if len(below_threshold_indices) > 0:
            cutoff_idx = below_threshold_indices[0]
            AARR_sparse[cutoff_idx:, i, :] = np.nan
            cutoff_x_pos = xpixF[cutoff_idx]
            print(f"  > ファイバー {i}: 基準輝度={reference_brightness:.1f} -> X={cutoff_x_pos}でトレース終了")

    # --- 4. 最終的なトレースの平滑化とファイル生成 ---
    print("\n--- フィルタリング後のデータで最終成果物を作成します ---")
    fppoly_data = np.full((num_fibers, num_disp_pts), np.nan, dtype=np.float64)
    pp1_data = np.zeros_like(data, dtype=np.int16)

    for i in iFibAct:
        polyfit_start, polyfit_end = config['polyfit_range_x']

        polyfit_mask = (xpixF >= polyfit_start) & (xpixF <= polyfit_end)
        valid_points_mask = np.isfinite(AARR_sparse[:, i, 1])
        final_mask = polyfit_mask & valid_points_mask

        if np.count_nonzero(final_mask) > config['min_points_for_polyfit']:
            x_fit = xpixF[final_mask]
            y_fit = AARR_sparse[final_mask, i, 1]

            coef = np.polyfit(x_fit, y_fit, poly_order)
            smooth_y_trace = np.polyval(coef, xpix)

            # 外挿領域をNaNでマスクして暴走を防ぐ
            min_fit_x, max_fit_x = np.min(x_fit), np.max(x_fit)
            extrapolation_mask = (xpix < min_fit_x) | (xpix > max_fit_x)
            smooth_y_trace[extrapolation_mask] = np.nan

            # 物理的にありえない値をクリッピング
            valid_trace_indices = ~np.isnan(smooth_y_trace)
            smooth_y_trace[valid_trace_indices] = np.clip(
                smooth_y_trace[valid_trace_indices], 0, num_spatial_pts - 1
            )
            fppoly_data[i, :] = smooth_y_trace

    # pp1.fits と fppoly.fits を生成・保存
    trace_width_radius = 1
    for i in iFibAct:
        y_trace = fppoly_data[i, :]
        for x_pos in range(num_disp_pts):
            y_pos_float = y_trace[x_pos]
            if not np.isnan(y_pos_float):
                y_pos = int(round(y_pos_float))
                y_min = max(0, y_pos - trace_width_radius)
                y_max = min(num_spatial_pts, y_pos + trace_width_radius + 1)
                pp1_data[y_min:y_max, x_pos] = i + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    file_pp1 = output_dir / data_filepath.name.replace(".fits", ".pp1_2.fits")
    fits.writeto(file_pp1, pp1_data, header, overwrite=True)
    print(f"視覚化マップを保存しました: {file_pp1}")

    file_fppoly = output_dir / data_filepath.name.replace(".fits", ".fppoly.fits")
    hdu_fppoly = fits.PrimaryHDU(data=fppoly_data)
    hdu_fppoly.header['NAXIS1'] = num_disp_pts
    hdu_fppoly.header['NAXIS2'] = num_fibers
    hdu_fppoly.header['CTYPE1'] = 'Wavelength'
    hdu_fppoly.header['CTYPE2'] = 'Fiber_ID (0-based)'
    fits.HDUList([hdu_fppoly]).writeto(file_fppoly, overwrite=True)
    print(f"高精度軌跡データを保存しました: {file_fppoly}")


if __name__ == "__main__":
    # ===================================================================
    # --- ユーザー設定 ---
    # ===================================================================
    date = '20250501'  # 例
    base_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/{date}")
    dark_file = base_dir / "dk01h_20s.sp.fits"  # 例
    input_file = output_dir / ' led01r_clf590n_ga7000fsp220_1_nhp_py.fits'  # 例

    config = {
        # --- ファイバーの基本情報 ---
        'nFibX': 10, 'nFibY': 12,
        'iFibInact': [6, 49, 69, 89, 94, 109, 117],
        'yFib0': 16.0, 'yFib1': 8.3, 'ypixFibWid': 4.0,

        # --- トレースの基本設定 ---
        'trace_interval_x': 16,
        'gauss_fit_radius': 4,

        # --- 多項式フィット（平滑化）の設定 ---
        'poly_fit_order': 3,
        # 多項式フィットに使う、信頼できる波長の範囲(X座標)
        'polyfit_range_x': (100, 2048),
        # 平滑化カーブを計算するために必要な、最低限の有効なデータ点の数
        'min_points_for_polyfit': 20,

        # --- 相対的閾値フィルタリングの設定 ---
        # 各ファイバーの「基準の明るさ」を計算するための波長範囲(X座標)
        'reference_brightness_range': (100, 700),
        # 「基準の明るさ」の何%まで暗くなったらトレースを打ち切るか (0.3 = 30%)
        'trace_cutoff_fraction': 0.1,
    }

    create_final_trace_products(input_file, dark_file, output_dir, config)