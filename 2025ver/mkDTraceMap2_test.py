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

    # --- ▼▼▼ プロット機能の準備 ▼▼▼ ---
    if make_plots:
        print("\n--- 可視化プロットを生成します ---")
        # 画像保存用ディレクトリを作成
        png_dir = output_dir / "plots" / data_filepath.name.replace(".fits", "").replace("_nhp_py", "")
        png_dir.mkdir(parents=True, exist_ok=True)

        # 概要プロット (初期位置の確認)
        fig_overview, ax_overview = plt.subplots(figsize=(12, 8))
        vmin, vmax = np.nanpercentile(data, [5, 98])
        ax_overview.imshow(data, cmap='gray', vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        for y_pos in yfibs[iFibAct]:
            ax_overview.axhline(y_pos, color='red', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_overview.set_title(f"Initial Fiber Positions on {data_filepath.name}")
        ax_overview.set_xlabel("Dispersion Axis (pixels)")
        ax_overview.set_ylabel("Spatial Axis (pixels)")
        overview_path = png_dir / "0_overview.png"
        plt.savefig(overview_path, bbox_inches='tight')
        plt.close(fig_overview)
        print(f"概要プロットを保存しました: {overview_path}")

        # 詳細プロット用の図と軸を準備
        pny, pnx = 5, 3  # 1ファイバーあたり3プロット (1D Fit, 2D Trace, Residuals)
        fig_detail, axes_detail = plt.subplots(pny, pnx, figsize=(12, 16), dpi=100)
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        plot_idx, fig_num = 0, 1

    # --- 2. 各ファイバーのトレース処理 ---
    for i in iFibAct:
        print(f"ファイバー {i:3d}/{num_fibers - 1} を追跡中...", end='\r')
        m2 = yfibs[i]
        for j in xpixF:
        #for j in reversed(xpixF):
            ypix1 = (np.arange(ypixFibWid) - ypixFibWid / 2 + 0.5 + m2).astype(int)
            if ypix1.min() < 0 or ypix1.max() >= num_spatial_pts: continue
            ydat1 = data[ypix1, j]

            window_mean = np.mean(ypix1)

            # 初期値と探索範囲を、窓の平均座標を基準に設定 (fibfitの方式)
            Aini = [np.max(ydat1), window_mean, ypixFibWid / 5.0]
            param_bounds = ([1, window_mean - ypixFibWid/2, 0.1],  # 探索範囲の下限
                            [np.inf, window_mean + ypixFibWid/2, 3])  # 探索範囲の上限
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
        fppoly_data[i, :] = np.clip(smooth_y_trace, 0, num_spatial_pts - 1)

        # --- ▼▼▼ 詳細プロットの描画 ▼▼▼ ---
        if make_plots:
            if plot_idx >= pny * pnx:  # 図がいっぱいになったら保存して次へ
                # 未使用の軸を非表示に
                for k in range(plot_idx, pny * pnx): axes_detail.flat[k].axis('off')
                detail_path = png_dir / f"{fig_num}.png"
                plt.savefig(detail_path, bbox_inches='tight')
                plt.close(fig_detail)
                print(f"詳細プロット {fig_num} を保存しました。")
                # 新しい図を準備
                fig_detail, axes_detail = plt.subplots(pny, pnx, figsize=(12, 16), dpi=100)
                plt.subplots_adjust(wspace=0.3, hspace=0.4)
                plot_idx, fig_num = 0, fig_num + 1

            # プロット1: 1Dガウスフィットの様子 (最初のフィット点)
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

            plot_idx += 3  # 3つのプロットを使ったのでインデックスを3増やす

    # --- ループ終了後の後始末 ---
    if make_plots and plot_idx > 0:  # 最後に残ったプロットを保存
        for k in range(plot_idx, pny * pnx): axes_detail.flat[k].axis('off')
        detail_path = png_dir / f"{fig_num}.png"
        plt.savefig(detail_path, bbox_inches='tight')
        plt.close(fig_detail)
        print(f"詳細プロット {fig_num} を保存しました。")

    print("\n--- トレース結果をFITSファイルに保存します ---")
    # (FITSファイル保存処理は変更なし)
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

    file_pp1 = output_dir / data_filepath.name.replace(".fits", ".pp1_test.fits").replace("_nhp_py", "")
    hdu_pp1 = fits.PrimaryHDU(data=pp1_data, header=header)
    hdu_pp1.writeto(file_pp1, overwrite=True)
    print(f"視覚化マップを保存しました: {file_pp1}")

    file_fppoly = output_dir / data_filepath.name.replace(".fits", ".fppoly_test.fits").replace("_nhp_py", "")
    hdu_fppoly = fits.PrimaryHDU(data=fppoly_data)
    hdu_fppoly.header['NAXIS1'] = num_disp_pts
    hdu_fppoly.header['NAXIS2'] = num_fibers
    hdu_fppoly.header['CTYPE1'] = 'Pixel'
    hdu_fppoly.header['CTYPE2'] = 'Fiber_ID (0-based)'
    hdu_fppoly.header['HISTORY'] = 'Traced with fibfit-like script (with plots).'
    hdu_fppoly.writeto(file_fppoly, overwrite=True)
    print(f"高精度軌跡データを保存しました: {file_fppoly}")


if __name__ == "__main__":
    # --- 基本設定 ---
    date = '20250501'
    base_output_dir = Path(f"C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/")
    output_dir = base_output_dir / date

    # --- ダークファイル（これは固定） ---
    dark_file = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/dk01h_20s.sp.fits")

    # --- 計算パラメータ ---
    config = {
        'nFibX': 10, 'nFibY': 12,
        'iFibInact': [6, 49, 69, 89, 94, 109, 117],
        'yFib0': 195, 'yFib1': 5.95,
        'ypixFibWid': 5.0,
        'trace_interval_x': 4,  # 精度を上げるため4に変更（お好みで調整）
        'poly_fit_order': 6,
        'use_two_pass_fit': True,
    }

    # --- ▼▼▼ ここからが新しい入り口 ▼▼▼ ---

    # ★★★ ここに、処理したいFITSファイルのフルパスを直接指定してください ★★★
    # (例: ホットピクセル除去後のLEDフラット画像など)
    direct_input_file = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20250501/led01r_sp.fits")

    print(f"--- 直接指定されたファイルで処理を開始します ---")
    print(f"入力ファイル: {direct_input_file}")

    if not direct_input_file.exists():
        print(f"エラー: 指定されたファイルが見つかりませんでした: {direct_input_file}")
    else:
        # トレース処理関数を直接呼び出す
        create_final_trace_products(
            data_filepath=direct_input_file,
            dark_filepath=dark_file,
            output_dir=output_dir,
            config=config,
            make_plots=True
        )

    print("\n--- トレース処理スクリプト完了 ---")

    # --- ▼▼▼ 元々のCSVから読み込む処理は、一旦コメントアウトします ▼▼▼
    #
    # csv_file_path = Path("mcparams202505.csv")
    # TRACE_TARGET_DESCRIPTION = 'LED'
    #
    # print(f"--- トレース処理の準備 ---")
    # try:
    #     df = pd.read_csv(csv_file_path)
    #     fits_col, desc_col = df.columns[0], df.columns[1]
    #     target_rows = df[df[desc_col] == TRACE_TARGET_DESCRIPTION]
    #     if target_rows.empty:
    #         print(f"エラー: CSV内に説明が '{TRACE_TARGET_DESCRIPTION}' のファイルが見つかりません。")
    #     else:
    #         original_fits_path_str = target_rows.iloc[0][fits_col]
    #         original_fits_path = Path(original_fits_path_str)
    #         nhp_fits_name = original_fits_path.stem + "_nhp_py.fits"
    #         input_file = output_dir / nhp_fits_name
    #         if not input_file.exists():
    #             print(f"エラー: トレース対象のファイルが見つかりませんでした: {input_file}")
    #         else:
    #             create_final_trace_products(input_file, dark_file, output_dir, config, make_plots=True)
    # except FileNotFoundError:
    #     print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
    # except (IndexError, KeyError):
    #     print("エラー: CSVファイルの形式が正しくないようです。")
    #
    # print("\n--- トレース処理スクリプト完了 ---")