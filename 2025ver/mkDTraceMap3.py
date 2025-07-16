import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from astropy.io import fits
from astropy.stats import sigma_clip  # <<< 変更点: sigma_clipをインポート
from pathlib import Path
import os


# --- ▼▼▼ 新設した関数 ▼▼▼ ---
def combine_fits_sigma_clip(file_list, sigma=3.0):
    """
    複数のFITSファイルをシグマクリップで合成する。
    """
    if not file_list:
        print("エラー: 合成対象のファイルリストが空です。")
        return None, None

    print(f"\n--- {len(file_list)}個のFITSファイルをシグマクリップで合成します ---")
    data_cube = []
    header = None

    for i, file_path in enumerate(file_list):
        try:
            with fits.open(file_path) as hdul:
                print(f"  読み込み中: {file_path.name}")
                # 最初のファイルのヘッダーを保持
                if i == 0:
                    header = hdul[0].header
                data_cube.append(hdul[0].data.astype(np.float64))
        except FileNotFoundError:
            print(f"  警告: ファイルが見つかりません {file_path}。スキップします。")
            continue

    if not data_cube:
        print("エラー: 有効なFITSファイルを読み込めませんでした。")
        return None, None

    # 3D配列に変換
    data_cube_np = np.array(data_cube)

    # シグマクリッピングを実行 (ファイル方向、つまりaxis=0でクリップ)
    print(f"シグマクリッピングを実行中 (sigma={sigma})...")
    clipped_cube = sigma_clip(data_cube_np, sigma=sigma, axis=0)

    # クリップ後のデータで平均を計算し、マスターフレームを作成
    print("マスターフレームを作成中...")
    master_frame = np.mean(clipped_cube, axis=0)

    # ヘッダー情報を更新
    if header:
        header['HISTORY'] = 'Combined with sigma clipping.'
        header['NCOMBINE'] = (len(data_cube), 'Number of combined frames')

    print("--- 合成完了 ---")
    return master_frame, header


def gaussian(x, amplitude, mean, stddev):
    """ガウス関数の定義"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


# --- ▼▼▼ 引数を変更した関数 ▼▼▼ ---
def create_final_trace_products(input_data, input_header, dark_filepath, output_dir, config, trace_source_name,
                                make_plots=True):
    """
    fibfit(mkFibFit4c)のロジックと可視化機能を取り入れたトレース処理。
    (ファイルパスの代わりにデータ配列を直接受け取るように修正)
    """
    print(f"--- 高精度トレース処理開始 (fibfit準拠・可視化機能付き) ---")
    print(f"データソース: {trace_source_name}")

    # --- 1. ファイル読み込みと前処理 ---
    # ファイル読み込み部分は削除し、引数のデータを直接使用
    data = input_data.copy().astype(np.float64)
    header = input_header

    if dark_filepath and dark_filepath.exists():
        with fits.open(dark_filepath) as hdul:
            data -= hdul[0].data

    data = median_filter(data, size=(1, 5))
    num_spatial_pts, num_disp_pts = data.shape

    # --- パラメータの展開 (変更なし) ---
    nFibX, nFibY = config['nFibX'], config['nFibY']
    iFibInact = config['iFibInact']
    yFib0, yFib1 = config['yFib0'], config['yFib1']
    ypixFibWid = config['ypixFibWid']
    finterval = config['trace_interval_x']
    poly_order = config['poly_fit_order']

    # --- 配列の初期化 (変更なし) ---
    iFib = np.arange(nFibX * nFibY)
    iFibAct = np.setdiff1d(iFib, iFibInact)
    yfibs = np.arange(len(iFib), dtype=float) * yFib1 + yFib0
    num_fibers = len(iFib)
    AARR = np.full((num_disp_pts, num_fibers, 3), np.nan, dtype=float)
    fppoly_data = np.full((num_fibers, num_disp_pts), np.nan, dtype=np.float64)
    xpix = np.arange(num_disp_pts)
    xpixF = np.arange(0, num_disp_pts, finterval)
    if xpixF.max() != (num_disp_pts - 1):
        xpixF = np.append(xpixF, num_disp_pts - 1)

    # --- ▼▼▼ プロット機能の準備 ▼▼▼ ---
    if make_plots:
        print("\n--- 可視化プロットを生成します ---")
        # 画像保存用ディレクトリを作成
        plot_name = Path(trace_source_name).stem
        png_dir = output_dir / "plots" / plot_name
        png_dir.mkdir(parents=True, exist_ok=True)

        # 概要プロット (初期位置の確認)
        fig_overview, ax_overview = plt.subplots(figsize=(12, 8))
        vmin, vmax = np.nanpercentile(data, [5, 98])
        ax_overview.imshow(data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        for y_pos in yfibs[iFibAct]:
            ax_overview.axhline(y_pos, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_overview.set_title(f"Initial Fiber Positions on {trace_source_name}")
        ax_overview.set_xlabel("Dispersion Axis (pixels)")
        ax_overview.set_ylabel("Spatial Axis (pixels)")
        overview_path = png_dir / "0_overview.png"
        plt.savefig(overview_path, bbox_inches='tight')
        plt.close(fig_overview)
        print(f"概要プロットを保存しました: {overview_path}")

        # 詳細プロット用の図と軸を準備
        pny, pnx = 5, 3
        fig_detail, axes_detail = plt.subplots(pny, pnx, figsize=(12, 16), dpi=100)
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        plot_idx, fig_num = 0, 1

    # --- 2. 各ファイバーのトレース処理 (以降、処理ロジックは変更なし) ---
    for i in iFibAct:
        print(f"ファイバー {i:3d}/{num_fibers - 1} を追跡中...", end='\r')
        m2 = yfibs[i]
        for j in xpixF:
            ypix1 = (np.arange(ypixFibWid) - ypixFibWid / 2 + 0.5 + m2).astype(int)
            if ypix1.min() < 0 or ypix1.max() >= num_spatial_pts: continue
            ydat1 = data[ypix1, j]
            window_mean = np.mean(ypix1)
            Aini = [np.max(ydat1), window_mean, ypixFibWid / 5.0]
            param_bounds = ([1, window_mean - ypixFibWid / 2, 0.1], [np.inf, window_mean + ypixFibWid / 2, 3])
            try:
                par, cov = curve_fit(gaussian, ypix1, ydat1, p0=Aini, bounds=param_bounds)
                m2 = par[1]
                AARR[j, i, :] = par
            except RuntimeError:
                continue

        valid_mask = np.isfinite(AARR[xpixF, i, 1])
        if np.count_nonzero(valid_mask) <= poly_order: continue
        x_sparse = xpixF[valid_mask]
        y_sparse = AARR[x_sparse, i, 1]
        coef_initial = np.polyfit(x_sparse, y_sparse, poly_order)
        residuals = y_sparse - np.polyval(coef_initial, x_sparse)
        sigma = np.std(residuals)
        inlier_mask = np.abs(residuals) <= 3.0 * sigma
        if np.count_nonzero(inlier_mask) > poly_order:
            coef_final = np.polyfit(x_sparse[inlier_mask], y_sparse[inlier_mask], poly_order)
        else:
            coef_final = coef_initial
        smooth_y_trace = np.polyval(coef_final, xpix)
        fppoly_data[i, :] = np.clip(smooth_y_trace, 0, num_spatial_pts - 1)

        if make_plots:
            if plot_idx >= pny * pnx:
                for k in range(plot_idx, pny * pnx): axes_detail.flat[k].axis('off')
                detail_path = png_dir / f"{fig_num}.png"
                plt.savefig(detail_path, bbox_inches='tight')
                plt.close(fig_detail)
                print(f"\n詳細プロット {fig_num} を保存しました。")
                fig_detail, axes_detail = plt.subplots(pny, pnx, figsize=(12, 16), dpi=100)
                plt.subplots_adjust(wspace=0.3, hspace=0.4)
                plot_idx, fig_num = 0, fig_num + 1

            ax1 = axes_detail.flat[plot_idx]
            j0 = xpixF[0]
            ypix1_plot = (np.arange(ypixFibWid) - ypixFibWid / 2 + 0.5 + yfibs[i]).astype(int)
            if not (ypix1_plot.min() < 0 or ypix1_plot.max() >= num_spatial_pts):
                ydat1_plot = data[ypix1_plot, j0]
                ax1.plot(ypix1_plot, ydat1_plot, 'o', markersize=3, label='Data')
                if np.isfinite(AARR[j0, i, 1]):
                    fit_curve = gaussian(ypix1_plot, *AARR[j0, i, :])
                    ax1.plot(ypix1_plot, fit_curve, '-', label='Fit')
            ax1.set_title(f"Fiber {i}: 1D Gaussian Fit @ X={j0}", fontsize=8)
            ax1.tick_params(axis='both', labelsize=6)
            ax2 = axes_detail.flat[plot_idx + 1]
            median_y = np.nanmedian(smooth_y_trace)
            roi_y_min, roi_y_max = max(0, int(median_y - ypixFibWid * 2)), min(num_spatial_pts,
                                                                               int(median_y + ypixFibWid * 2))
            if roi_y_min < roi_y_max:
                vmin, vmax = np.nanpercentile(data[roi_y_min:roi_y_max, :], [5, 98])
                ax2.imshow(data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
                ax2.plot(xpix, smooth_y_trace, 'r-', linewidth=1, label='Trace')
                ax2.set_ylim(roi_y_min, roi_y_max)
                ax2.set_xlim(0, num_disp_pts)
            ax2.set_title(f"Fiber {i}: 2D Trace", fontsize=8)
            ax2.tick_params(axis='both', labelsize=6)
            ax3 = axes_detail.flat[plot_idx + 2]
            ax3.plot(x_sparse, residuals, 'o', markersize=2)
            ax3.axhline(0, color='r', linestyle='--')
            ax3.axhline(3 * sigma, color='g', linestyle=':', label='3 sigma')
            ax3.axhline(-3 * sigma, color='g', linestyle=':')
            ax3.set_title(f"Fiber {i}: Polyfit Residuals (std={sigma:.2f})", fontsize=8)
            ax3.set_xlabel("Pixel", fontsize=7);
            ax3.set_ylabel("Residual", fontsize=7)
            ax3.tick_params(axis='both', labelsize=6)
            plot_idx += 3

    if make_plots and plot_idx > 0:
        for k in range(plot_idx, pny * pnx): axes_detail.flat[k].axis('off')
        detail_path = png_dir / f"{fig_num}.png"
        plt.savefig(detail_path, bbox_inches='tight')
        plt.close(fig_detail)
        print(f"\n詳細プロット {fig_num} を保存しました。")

    print("\n--- トレース結果をFITSファイルに保存します ---")
    pp1_data = np.zeros_like(data, dtype=np.int16)
    trace_width_radius = 1
    for i in iFibAct:
        y_trace = fppoly_data[i, :]
        valid_trace = np.isfinite(y_trace)
        for x_pos in xpix[valid_trace]:
            y_pos = int(round(y_trace[x_pos]))
            y_min, y_max = max(0, y_pos - trace_width_radius), min(num_spatial_pts, y_pos + trace_width_radius + 1)
            pp1_data[y_min:y_max, x_pos] = i + 1

    file_pp1 = output_dir / f"{plot_name}.pp1_test.fits"
    hdu_pp1 = fits.PrimaryHDU(data=pp1_data, header=header)
    hdu_pp1.writeto(file_pp1, overwrite=True)
    print(f"視覚化マップを保存しました: {file_pp1}")

    file_fppoly = output_dir / f"{plot_name}.fppoly_test.fits"
    hdu_fppoly = fits.PrimaryHDU(data=fppoly_data)
    hdu_fppoly.header.update(header)
    hdu_fppoly.header['NAXIS1'] = num_disp_pts
    hdu_fppoly.header['NAXIS2'] = num_fibers
    hdu_fppoly.header['CTYPE1'] = 'Pixel'
    hdu_fppoly.header['CTYPE2'] = 'Fiber_ID (0-based)'
    hdu_fppoly.header['HISTORY'] = 'Traced with combined master LED frame.'
    hdu_fppoly.writeto(file_fppoly, overwrite=True)
    print(f"高精度軌跡データを保存しました: {file_fppoly}")


# --- ▼▼▼ 実行ブロックを大幅に修正 ▼▼▼ ---
if __name__ == "__main__":
    date = '20250501'
    base_output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/")
    output_dir = base_output_dir / date
    csv_file_path = Path("mcparams202505.csv")
    TRACE_TARGET_DESCRIPTION = 'LED'
    dark_file = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/dk01h_20s.sp.fits")
    config = {
        'nFibX': 10, 'nFibY': 12,
        'iFibInact': [6, 49, 69, 89, 94, 109, 117],
        'yFib0': 195, 'yFib1': 5.95, 'ypixFibWid': 5.0,
        'trace_interval_x': 4,
        'poly_fit_order': 6,
    }

    print(f"--- トレース処理の準備 ---")
    try:
        df = pd.read_csv(csv_file_path)
        fits_col, desc_col = df.columns[0], df.columns[1]

        # 1. CSVから 'LED' の全ファイルパスをリストアップ
        led_rows = df[df[desc_col] == TRACE_TARGET_DESCRIPTION]
        if led_rows.empty:
            raise FileNotFoundError(f"エラー: CSV内に説明が '{TRACE_TARGET_DESCRIPTION}' のファイルが見つかりません。")

        led_files_to_combine = []
        for index, row in led_rows.iterrows():
            original_fits_path = Path(row[fits_col])
            # トレース対象は _nhp_py.fits ファイル
            nhp_fits_name = original_fits_path.stem + "_nhp_py.fits"
            input_file_path = output_dir / nhp_fits_name
            if input_file_path.exists():
                led_files_to_combine.append(input_file_path)
            else:
                print(f"警告: トレース対象のファイルが見つかりません: {input_file_path}。スキップします。")

        # 2. シグマクリップでマスターLEDフレームを作成
        master_led_data, master_led_header = combine_fits_sigma_clip(led_files_to_combine)

        # 3. マスターフレームを使ってトレースを実行
        if master_led_data is not None:
            # トレース結果のファイル名やプロット名として使用
            trace_source_name = "master_led_trace"
            create_final_trace_products(
                master_led_data,
                master_led_header,
                dark_file,
                output_dir,
                config,
                trace_source_name,
                make_plots=True
            )
        else:
            print("エラー: マスターLEDフレームの作成に失敗したため、トレース処理を中止します。")

    except FileNotFoundError as e:
        print(e)
    except (IndexError, KeyError) as e:
        print(f"エラー: CSVファイルの形式が正しくないようです。 {e}")

    print("\n--- トレース処理スクリプト完了 ---")