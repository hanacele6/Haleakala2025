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


def create_final_trace_products(data_filepath, dark_filepath, output_dir, config, make_plots=True):
    """
    fibfit(mkFibFit4c)のロジックと可視化機能を取り入れたトレース処理。
    「累積更新」ロジックを実装済み。
    """
    print(f"--- 高精度トレース処理開始 (fibfit準拠・可視化機能付き) ---")
    print(f"データファイル: {data_filepath.name}")

    # --- 1. ファイル読み込みと前処理 ---
    try:
        with fits.open(data_filepath) as hdul:
            data = hdul[0].data.astype(np.float64)
            header = hdul[0].header
        if dark_filepath and dark_filepath.exists():
            with fits.open(dark_filepath) as hdul:
                data -= hdul[0].data
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return

    data = median_filter(data, size=(1, 5))
    num_spatial_pts, num_disp_pts = data.shape

    # --- パラメータの展開 ---
    nFibX, nFibY = config['nFibX'], config['nFibY']
    iFibInact = config['iFibInact']
    yFib0, yFib1 = config['yFib0'], config['yFib1']
    ypixFibWid = config['ypixFibWid']
    finterval = config['trace_interval_x']
    poly_order = config['poly_fit_order']

    # --- 配列の初期化 ---
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

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # 元の「全体的な位置ズレを自己補正」ブロックは削除します。
    # 新しいロジックは下のメインループ内に組み込まれています。
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    # --- ▼▼▼ プロット機能の準備 ▼▼▼ ---
    if make_plots:
        print("\n--- 可視化プロットを生成します ---")
        png_dir = output_dir / "plots" / data_filepath.name.replace(".fits", "").replace("_nhp_py", "")
        png_dir.mkdir(parents=True, exist_ok=True)

        # 概要プロットは一度だけ作成
        fig_overview, ax_overview = plt.subplots(figsize=(12, 8))
        vmin, vmax = np.nanpercentile(data, [5, 98])
        ax_overview.imshow(data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        # この時点でのyfibs（理論値）をプロット
        for y_pos in yfibs[iFibAct]:
            ax_overview.axhline(y_pos, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_overview.set_title(f"Initial Theoretical Fiber Positions on {data_filepath.name}")
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

    # --- 2. 各ファイバーのトレース処理 ---
    for i in iFibAct:
        print(f"ファイバー {i:3d}/{num_fibers - 1} を追跡中...", end='\r')

        # ★★★ ここからが元コードの「累積更新」を再現する核心部分 ★★★
        # (1) 現在のファイバーの最初の点(j0)でフィットを行い、予測位置とのズレを計算する
        j0 = xpixF[0]
        m2_for_correction = yfibs[i]
        ypix1_corr = (np.arange(ypixFibWid) - ypixFibWid / 2 + 0.5 + m2_for_correction).astype(int)

        if not (ypix1_corr.min() < 0 or ypix1_corr.max() >= num_spatial_pts):
            ydat1_corr = data[ypix1_corr, j0]
            try:
                window_mean_corr = np.mean(ypix1_corr)
                Aini_corr = [np.max(ydat1_corr), window_mean_corr, ypixFibWid / 5.0]
                param_bounds_corr = ([1, window_mean_corr - ypixFibWid / 2, 0.1],
                                     [1e5, window_mean_corr + ypixFibWid / 2, 3])
                par_corr, _ = curve_fit(gaussian, ypix1_corr, ydat1_corr,
                                        p0=Aini_corr, bounds=param_bounds_corr, max_nfev=2000)

                # (2) ズレ(dlt)を計算し、yfibs配列全体を更新する
                dlt = par_corr[1] - yfibs[i]
                yfibs += dlt  # <<< これが「累積更新」。次のファイバーの予測位置が補正される。

            except RuntimeError:
                # フィットに失敗した場合はyfibsを更新しない
                print(f"警告: ファイバー{i}の初期位置補正フィットに失敗。yfibsを更新しません。")
        # ★★★ ここまでが核心部分 ★★★

        # (3) 更新されたyfibsを元に、このファイバーのトレースを開始する
        m2 = yfibs[i]
        for j in xpixF:
            ypix1 = (np.arange(ypixFibWid) - ypixFibWid / 2 + 0.5 + m2).astype(int)
            if ypix1.min() < 0 or ypix1.max() >= num_spatial_pts: continue
            ydat1 = data[ypix1, j]

            window_mean = np.mean(ypix1)
            Aini = [np.max(ydat1), window_mean, ypixFibWid / 5.0]
            param_bounds = ([1, window_mean - ypixFibWid / 2, 0.1],
                            [1e5, window_mean + ypixFibWid / 2, 3])
            try:
                par, cov = curve_fit(gaussian, ypix1, ydat1, p0=Aini, bounds=param_bounds)
                m2 = par[1]
                AARR[j, i, :] = par
            except RuntimeError:
                continue

        # --- 3. 多項式フィット & シグマクリッピング ---
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

        # 元コードの挙動を再現するため、AARRを上書き
        AARR[xpix, i, 1] = smooth_y_trace
        fppoly_data[i, :] = np.clip(AARR[xpix, i, 1], 0, num_spatial_pts - 1)

        # --- ▼▼▼ 詳細プロットの描画 ▼▼▼ ---
        if make_plots:
            if plot_idx >= pny * pnx:
                for k in range(plot_idx, pny * pnx): axes_detail.flat[k].axis('off')
                detail_path = png_dir / f"{fig_num}.png"
                plt.savefig(detail_path, bbox_inches='tight')
                plt.close(fig_detail)
                print(f"詳細プロット {fig_num} を保存しました。")
                fig_detail, axes_detail = plt.subplots(pny, pnx, figsize=(12, 16), dpi=100)
                plt.subplots_adjust(wspace=0.3, hspace=0.4)
                plot_idx, fig_num = 0, fig_num + 1

            # プロット1: 1Dガウスフィットの様子 (最初のフィット点)
            # このプロットでは、yfibs[i]が更新された後の値を使うため、緑の線がピークに合う
            ax1 = axes_detail.flat[plot_idx]
            j0_plot = xpixF[0]
            ypix0_plot = (np.arange(ypixFibWid + 15) - ypixFibWid / 2 - 7 + yfibs[i]).astype(int)
            ypix1_plot = (np.arange(ypixFibWid) - ypixFibWid / 2 + 0.5 + yfibs[i]).astype(int)
            if not (ypix0_plot.min() < 0 or ypix0_plot.max() >= num_spatial_pts or \
                    ypix1_plot.min() < 0 or ypix1_plot.max() >= num_spatial_pts):
                ydat0_plot = data[ypix0_plot, j0_plot]
                ydat1_plot = data[ypix1_plot, j0_plot]
                ax1.plot(ypix0_plot, ydat0_plot, linewidth=1, color='black')
                ax1.plot(ypix1_plot, ydat1_plot, linewidth=2, color='black')
                ax1.plot(ypix1_plot, ydat1_plot, 'o', color='blue', markersize=3)
                if np.isfinite(AARR[j0_plot, i, 1]):
                    fit_curve = gaussian(ypix1_plot, *AARR[j0_plot, i, :])
                    ax1.plot(ypix1_plot, fit_curve, '-', color='red', linewidth=2, label='Fit')
                ax1.axvline(x=yfibs[i], color='green', linestyle='--')
                if i > 0 and i - 1 in iFibAct:
                    ax1.axvline(x=yfibs[i - 1], color='green', linestyle=':', linewidth=0.8)
                if i < (len(yfibs) - 1) and i + 1 in iFibAct:
                    ax1.axvline(x=yfibs[i + 1], color='green', linestyle=':', linewidth=0.8)
            fit_center_val = AARR[j0_plot, i, 1] if np.isfinite(AARR[j0_plot, i, 1]) else np.nan
            ax1.set_title(f'iFib={i} j={j0_plot} center={fit_center_val:.2f}', fontsize=6)
            ax1.tick_params(axis='both', labelsize=6)

            # プロット2: 2D画像上のトレース結果
            ax2 = axes_detail.flat[plot_idx + 1]
            median_y = np.nanmedian(smooth_y_trace)
            roi_y_min = max(0, int(median_y - ypixFibWid * 2))
            roi_y_max = min(num_spatial_pts, int(median_y + ypixFibWid * 2))
            if roi_y_min < roi_y_max:
                vmin, vmax = np.nanpercentile(data[roi_y_min:roi_y_max, :], [5, 98])
                ax2.imshow(data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
                ax2.plot(xpix, smooth_y_trace, 'r-', linewidth=1, label='Trace')
                ax2.set_ylim(roi_y_min, roi_y_max)
                ax2.set_xlim(0, num_disp_pts)
            ax2.set_title(f"Fiber {i}: 2D Trace", fontsize=8)
            ax2.tick_params(axis='both', labelsize=6)

            # プロット3: 多項式フィットの残差
            ax3 = axes_detail.flat[plot_idx + 2]
            ax3.plot(x_sparse, residuals, 'o', markersize=2)
            ax3.axhline(0, color='r', linestyle='--')
            ax3.axhline(3 * sigma, color='g', linestyle=':', label='3 sigma')
            ax3.axhline(-3 * sigma, color='g', linestyle=':')
            ax3.set_title(f"Fiber {i}: Polyfit Residuals (std={sigma:.2f})", fontsize=8)
            ax3.set_xlabel("Pixel", fontsize=7)
            ax3.set_ylabel("Residual", fontsize=7)
            ax3.tick_params(axis='both', labelsize=6)

            plot_idx += 3

    # --- ループ終了後の後始末 ---
    if make_plots and plot_idx > 0:
        for k in range(plot_idx, pny * pnx): axes_detail.flat[k].axis('off')
        detail_path = png_dir / f"{fig_num}.png"
        plt.savefig(detail_path, bbox_inches='tight')
        plt.close(fig_detail)
        print(f"詳細プロット {fig_num} を保存しました。")

    print("\n--- トレース結果をFITSファイルに保存します ---")
    pp1_data = np.zeros_like(data, dtype=np.int16)
    trace_width_radius = 1
    for i in iFibAct:
        y_trace = fppoly_data[i, :]
        valid_trace = np.isfinite(y_trace)
        for x_pos in xpix[valid_trace]:
            y_pos = int(round(y_trace[x_pos]))
            y_min = max(0, y_pos - trace_width_radius)
            y_max = min(num_spatial_pts, y_pos + trace_width_radius + 1)
            pp1_data[y_min:y_max, x_pos] = i + 1

    file_pp1 = output_dir / data_filepath.name.replace(".fits", ".pp1.fits").replace("_nhp_py", "")
    hdu_pp1 = fits.PrimaryHDU(data=pp1_data, header=header)
    hdu_pp1.writeto(file_pp1, overwrite=True)
    print(f"視覚化マップを保存しました: {file_pp1}")

    file_fppoly = output_dir / data_filepath.name.replace(".fits", ".fppoly.fits").replace("_nhp_py", "")
    hdu_fppoly = fits.PrimaryHDU(data=fppoly_data)
    hdu_fppoly.header['NAXIS1'] = num_disp_pts
    hdu_fppoly.header['NAXIS2'] = num_fibers
    hdu_fppoly.header['CTYPE1'] = 'Pixel'
    hdu_fppoly.header['CTYPE2'] = 'Fiber_ID (0-based)'
    hdu_fppoly.header['HISTORY'] = 'Traced with fibfit-like script (with plots).'
    hdu_fppoly.writeto(file_fppoly, overwrite=True)
    print(f"高精度軌跡データを保存しました: {file_fppoly}")


if __name__ == "__main__":
    date = '20250710'
    base_output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/")
    output_dir = base_output_dir / date
    csv_file_path = Path("mcparams20250710.csv") # 忘れずに！！！！
    TRACE_TARGET_DESCRIPTION = 'LED'
    dark_file = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/dk01h_20s.sp.fits")
    config = {
        'nFibX': 10, 'nFibY': 12,
        'iFibInact': [6, 49, 69, 89, 94, 109, 117],  # 使ってないファイバー
        'yFib0': 195, 'yFib1': 5.95, 'ypixFibWid': 6.0,
        'trace_interval_x': 1,  # fitを行う間隔
        'poly_fit_order': 6,
    }

    print(f"--- トレース処理の準備 ---")
    try:
        df = pd.read_csv(csv_file_path)
        fits_col, desc_col = df.columns[0], df.columns[1]
        target_rows = df[df[desc_col] == TRACE_TARGET_DESCRIPTION]
        if target_rows.empty:
            print(f"エラー: CSV内に説明が '{TRACE_TARGET_DESCRIPTION}' のファイルが見つかりません。")
        else:
            original_fits_path_str = target_rows.iloc[0][fits_col]
            original_fits_path = Path(original_fits_path_str)
            nhp_fits_name = original_fits_path.stem + "_nhp_py.fits"
            input_file = output_dir / nhp_fits_name
            if not input_file.exists():
                print(f"エラー: トレース対象のファイルが見つかりませんでした: {input_file}")
            else:
                create_final_trace_products(input_file, dark_file, output_dir, config, make_plots=True)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
    except (IndexError, KeyError):
        print("エラー: CSVファイルの形式が正しくないようです。")

    print("\n--- トレース処理スクリプト完了 ---")