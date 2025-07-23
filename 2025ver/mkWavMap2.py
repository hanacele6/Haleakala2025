import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import convolve
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.stats import sigma_clip
import pandas as pd
import os
from pathlib import Path
import sys

# ==============================================================================
# 補助関数の定義
# ==============================================================================
plt.rcParams.update({'font.size': 8})


def combine_fits_clipped_mean(file_list, output_file):
    """
    複数のFITSファイルをシグマクリッピングしながら平均合成する関数。
    """
    dcb = []
    header0 = None

    print("\n" + "=" * 60)
    print(f"ステップ1: {len(file_list)} 個のSKYフレームをマスターフレームに合成します...")
    for f in file_list:
        print(f"  - 読み込み中: {os.path.basename(f)}")
        with fits.open(f) as hdul:
            if header0 is None:
                header0 = hdul[0].header
            dcb.append(hdul[0].data.astype(np.float64))

    dcb_array = np.array(dcb)

    print("  -> シグマクリッピングを実行中...")
    clipped_data = sigma_clip(dcb_array, sigma=2.0, axis=0, maxiters=3)

    master_frame = np.ma.mean(clipped_data, axis=0).data
    print("  -> 合成が完了しました。")

    header0['HISTORY'] = 'Combined with sigma clipping'
    header0['NCOMBINE'] = (len(file_list), 'Number of combined frames')

    hdu = fits.PrimaryHDU(data=master_frame.astype(np.float32), header=header0)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_file, overwrite=True)
    print(f"  -> マスターフレームを保存しました: {output_file}")
    return output_file


def residuals4(A, x, y):
    """ガウス関数と実データの残差"""
    z = (x - A[1]) / A[2]
    model = A[0] * np.exp(-(z ** 2) / 2) + A[3]
    return model - y


def gaussian_kernel(size, sigma=1):
    """ガウシアンカーネルを生成する"""
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


# ==============================================================================
# メインの波長校正関数
# ==============================================================================
def mkWavMap4b(input_fits_path, solar_spec_path, params,flgPause=True):
    # --- 1. ファイルの存在確認と読み込み ---
    base_filename = os.path.basename(input_fits_path)
    output_dir = os.path.dirname(input_fits_path)
    fileWMP = os.path.join(output_dir, base_filename.replace("_f.fits", ".wmp.fits"))

    print("\n" + "=" * 60)
    print(f"ステップ2: 波長校正を開始します: {base_filename}")

    if not os.path.exists(input_fits_path):
        print(f"  -> エラー: ファイルが存在しません。処理を終了します。")
        return

    try:
        with fits.open(input_fits_path) as hdul:
            spDat = hdul[0].data.astype(np.float64)
            hd = hdul[0].header
            ny, nx = spDat.shape
            iFibAct = np.arange(ny)
    except Exception as e:
        print(f"  -> ファイル読み込みエラー: {e}")
        return

    # --- 2. データの前処理 ---
    print("  -> シグマクリッピングとメディアンフィルタを適用...")
    for j in iFibAct:
        clipped_spec = sigma_clip(spDat[j, :], sigma=3, maxiters=5, masked=True)
        spDat[j, clipped_spec.mask] = np.ma.median(clipped_spec)
    for iJ in range(ny):
        spDat[iJ, :] = median_filter(spDat[iJ, :], size=3)

    # --- 3. 参照スペクトルの準備 ---
    pxlinesD0 = params['pxlinesD0_base'] + params['pixdwavs_val']
    xpix = np.arange(nx)

    if (params['calwav1'] >= 586 and params['calwav1'] <= 596):
        fileCal = 'psgrad586-596.txt'
    else:
        print(f"エラー: 波長域 {params['calwav1']} nm に対応する参照スペクトルが設定されていません。")
        return

    try:
        spMdl = np.loadtxt(os.path.join(solar_spec_path, fileCal), skiprows=14)
    except FileNotFoundError:
        print(f"  -> エラー: 参照スペクトルが見つかりません: {os.path.join(solar_spec_path, fileCal)}")
        return

    # --- 4. 参照スペクトルから精密な波長を決定 ---
    print("  -> 参照スペクトルから精密な輝線波長を決定...")
    wavs = np.zeros_like(params['wlinesM'])
    #wavair2vac = 1.000276
    wavair2vac = 1.000000
    kernel = gaussian_kernel(size=181, sigma=61)
    xm = spMdl[:, 0] / wavair2vac
    cv = convolve(spMdl[:, 1], kernel, mode='same')
    ym = cv / np.median(cv)


    for idx, w_approx in enumerate(params['wlinesM']):
        xrng_wav = w_approx + np.array([-0.15, 0.15])
        ixm = (xm >= xrng_wav[0]) & (xm <= xrng_wav[1])
        if ixm.sum() < 5: continue
        Aini = [np.min(ym[ixm]) - 1, w_approx, 0.02, 1.0]
        res = least_squares(residuals4, Aini, args=(xm[ixm], ym[ixm]), loss='huber')
        wavs[idx] = res.x[1]
    print(f"  -> 精密な輝線波長 [nm]: {np.round(wavs, 4)}")

    # --- 5. 波長校正の実行 ---
    print("  -> 波長校正のフィッティングを開始...")
    wmp = np.zeros_like(spDat)
    y0even = pxlinesD0[0]
    y0odd = y0even + params['dltFibY']

    fig_detail, axes_detail = plt.subplots(4, 4, figsize=(12, 10), dpi=96)
    axes_detail_flat = axes_detail.flatten()

    #fig_detail, axes_detail = plt.subplots(7, 4, figsize=(12, 16), dpi=96)
    #plt.subplots_adjust(wspace=0.5, hspace=0.8, left=0.08, right=0.95, top=0.95, bottom=0.05)
    #iFig = 0


    for j in iFibAct:

        print(f"  -> Processing Fiber {j:03d}...")
        iFig = 0
        for ax in axes_detail_flat: ax.clear()

        if j % 2 == 0:
            pxlinesD_current = pxlinesD0 - pxlinesD0[0] + y0even
        else:
            pxlinesD_current = pxlinesD0 - pxlinesD0[0] + y0odd

        wpix_fit = np.full_like(pxlinesD_current, np.nan, dtype=np.float64)

        for idx, p_approx in enumerate(pxlinesD_current):
            # ★★★★★ 元のコードの2段階フィットロジックを完全再現 ★★★★★
            # ステップ1: 広い範囲で大まかに中心を探す
            xrng1 = p_approx + np.array([-0.5, 0.5]) * 0.3 / params['wavstep1']
            ixd1 = (xpix >= max(0, xrng1[0])) & (xpix < min(nx, xrng1[1]))
            xd1 = xpix[ixd1]
            yd1_raw = spDat[j, ixd1]
            if yd1_raw.size < 5: continue

            max_val1 = np.max(yd1_raw)
            if max_val1 == 0: continue
            yd1_norm = yd1_raw / max_val1

            Aini1 = [-0.8, p_approx, 3.0, 1.0]
            res1 = least_squares(residuals4, Aini1, args=(xd1, yd1_norm), loss='huber')
            center_approx = res1.x[1]

            # ステップ2: 狭い範囲で精密にフィットする
            xrng2 = center_approx + np.array([-0.5, 0.5]) * 0.07 / params['wavstep1']
            ixd2 = (xpix >= max(0, xrng2[0])) & (xpix < min(nx, xrng2[1]))
            xd2 = xpix[ixd2]
            yd2_raw = spDat[j, ixd2]
            if yd2_raw.size < 5: continue

            max_val2 = np.max(yd2_raw)
            if max_val2 == 0: continue
            yd2_norm = yd2_raw / max_val2

            Aini2 = [-0.8, center_approx, 3.0, 1.0]
            bounds2 = ([-1.5, center_approx - 5, 1.0, 0.5], [0.0, center_approx + 5, 8.0, 1.5])
            res2 = least_squares(residuals4, Aini2, args=(xd2, yd2_norm), loss='huber', bounds=bounds2)

            wpix_fit[idx] = res2.x[1]


            if iFig < len(axes_detail_flat):
                ax = axes_detail_flat[iFig]
                ax.plot(xpix, spDat[j, :] / np.median(spDat[j, :]), linewidth=1, color='gray')  # 全体像
                ax.plot(xd2, yd2_raw / np.median(spDat[j, :]), '.-', label='data')  # 狭い範囲のデータ
                fit_y = residuals4(res2.x, xd2, 0) * max_val2 / np.median(spDat[j, :])
                ax.plot(xd2, fit_y, '--', label='fit')
                ax.set_title(f"L{idx}: {p_approx:.1f}->{res2.x[1]:.1f}", fontsize=7)
                ax.axvline(res2.x[1], color='r', linestyle='--', linewidth=1)
                ax.set_xlim(xrng1)
                ax.tick_params(labelsize=6)
                if idx == 0: ax.legend(fontsize='xx-small')
                iFig += 1

        valid_indices = ~np.isnan(wpix_fit)
        ndeg = 5
        if np.sum(valid_indices) < ndeg + 1:
            print(f"  -> 警告: Fiber {j} で有効なフィット点が{ndeg + 1}未満のため、スキップします。")
            wmp[j, :] = np.nan
            continue

        coef = np.polyfit(wpix_fit[valid_indices], wavs[valid_indices], ndeg)
        pfit = np.poly1d(coef)

        max_residual_pm = 3.0
        residuals_nm = wavs[valid_indices] - pfit(wpix_fit[valid_indices])
        mask = np.abs(residuals_nm * 1000) <= max_residual_pm

        if np.any(~mask):
            wpix_good = wpix_fit[valid_indices][mask]
            wavs_good = wavs[valid_indices][mask]
            if len(wpix_good) >= ndeg + 1:
                coef = np.polyfit(wpix_good, wavs_good, ndeg)
                pfit = np.poly1d(coef)

        wmp[j, :] = pfit(xpix)

        ifunc = interp1d(wmp[j, :], xpix, kind='linear', fill_value='extrapolate', bounds_error=False)
        xpix0 = ifunc(wavs[0])

        if np.isnan(xpix0):
            print(f"  -> 警告: Fiber {j} でxpix0の計算に失敗しました。追跡をリセットします。")
            y0even = pxlinesD0[0]
            y0odd = y0even + params['dltFibY']
        elif j % 2 == 0:
            y0even = xpix0;
            y0odd = y0even + params['dltFibY']
        else:
            y0odd = xpix0;
            y0even = y0odd - params['dltFibY']

        plt.suptitle(f"Fitting Details for Fiber {j}", fontsize=16)
        plt.draw()
        plt.pause(0.01)  # このpauseが描画に必要
        if flgPause: input(f"Fiber {j} のフィット結果です。Enterで続行...")

    print(f"  -> フィッティング完了。")

    # --- 6. 結果の保存 ---
    hd_out = hd.copy()
    hd_out['HISTORY'] = 'Wavelength calibrated'
    hd_out['BUNIT'] = ('nm', 'Wavelength unit')

    hdu = fits.PrimaryHDU(data=wmp.astype(np.float32), header=hd_out)
    hdul = fits.HDUList([hdu])
    os.makedirs(os.path.dirname(fileWMP), exist_ok=True)
    hdul.writeto(fileWMP, overwrite=True)
    print(f"  -> 波長マップを保存しました: {os.path.basename(fileWMP)}")


# ==============================================================================
# スクリプトの実行
# ==============================================================================
if __name__ == "__main__":
    # --- 設定項目 ---
    base_dir = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/"
    csv_path = Path("mcparams202505.csv")
    output_dir = os.path.join(base_dir, "output/20250501/")
    master_sky_filename = "master_sky_f.fits"

    # 波長校正パラメータ（固定値）
    calibration_params = {
        'wlinesM': np.array([588.544, 589.157, 589.449, 589.756, 590.732, 591.580, 591.789]),
        'pxlinesD0_base': np.array([769, 949, 1035, 1127, 1423, 1557, 1685]),
        'pixdwavs_val': 0.0,
        'pixwav1': 1024,
        'wavstep1': 0.00293,
        'calwav1': 588.9,
        'dltFibY': -12.5
    }
    # --- 設定ここまで ---

    ### ステップ1: CSVからSKYフレームのリストを取得し、マスターSKYを作成 ###
    try:
        df = pd.read_csv(csv_path)
        df_sky = df[df['Type'].str.strip().str.upper() == 'SKY'].copy()

        sky_file_paths = []
        sky_file_paths = []
        # enumerateを使って、SKYタイプのファイルが何番目に出てきたかを取得
        for i, (index, row) in enumerate(df_sky.iterrows()):
            # iは0から始まるので、ファイル番号はi+1とする
            # 例: 1番目のSKYファイル -> SKY1_f.fits
            correct_filename = f"{row['Type'].upper()}{i + 1}_f.fits"
            correct_filepath = os.path.join(output_dir, correct_filename)
            sky_file_paths.append(correct_filepath)
    except Exception as e:
        print(f"CSVファイルの読み込みまたは解析中にエラーが発生しました: {e}")
        sys.exit()

    if not sky_file_paths:
        print("エラー: CSV内にSKYタイプのファイルが見つかりません。")
    else:
        master_sky_filepath = os.path.join(output_dir, master_sky_filename)
        os.makedirs(output_dir, exist_ok=True)
        # マスターSKYフレームを作成
        combine_fits_clipped_mean(sky_file_paths, master_sky_filepath)

        ### ステップ2: 作成したマスターSKYの波長校正を実行 ###
        # 波長校正関数を呼び出し
        mkWavMap4b(master_sky_filepath, os.path.join(base_dir, "psg/"), calibration_params,flgPause=True)

        print("\n" + "=" * 60)
        print("すべての処理が完了しました。")