### Last update on 18-JUL-2025 (ロジック修正版)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import convolve
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from astropy.io import fits
import os
import sys

# ==============================================================================
# 補助関数の定義
# ==============================================================================
plt.rcParams.update({'font.size': 8})
plt.rcParams['axes.unicode_minus'] = False


def gauss1d4(x, A):
    """1次元ガウス関数を定義します（ベースライン有り）。"""
    z = (x - A[1]) / A[2]
    return A[0] * np.exp(-(z ** 2) / 2) + A[3]


def residuals4(A, x, y):
    """データとガウスモデルの残差を計算します。"""
    return gauss1d4(x, A) - y


def gaussian_kernel(size, sigma=1):
    """畳み込み用のガウシアンカーネルを生成します。"""
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


# ==============================================================================
# メインの波長校正関数
# ==============================================================================
def mkWavMap4b_final(input_fsp_path, solar_spec_dir, params, flgPause=True, save_fiber_plots=True):
    """2次元スペクトルFITSファイルの波長校正を実行します。"""
    base_filename = os.path.basename(input_fsp_path)
    output_dir = os.path.dirname(input_fsp_path)
    fileWMP = os.path.join(output_dir, base_filename.replace(".fits", ".wmp.fits"))

    plot_output_dir = os.path.join(output_dir, "fiber_plots", os.path.splitext(base_filename)[0])
    if save_fiber_plots:
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"  -> Plots for each fiber will be saved to '{plot_output_dir}'")

    print("\n" + "=" * 80)
    print(f"Starting Wavelength Calibration for: {base_filename}")
    print(f"Output file: {fileWMP}")
    print("=" * 80)

    # --- ファイル読み込み ---
    try:
        with fits.open(input_fsp_path) as hdul:
            spDat = hdul[0].data.astype(np.float64)
            hd = hdul[0].header
            ny, nx = spDat.shape
            try:
                iFibAct = hdul['IFIBERS'].data
                print(f"  -> Found 'IFIBERS' extension. Processing {len(iFibAct)} active fibers.")
            except KeyError:
                iFibAct = np.arange(ny)
                print("  -> 'IFIBERS' extension not found. Processing all fibers.")
    except Exception as e:
        print(f"  -> ERROR: Could not read input FITS file: {e}");
        return

    # --- パラメータ展開 ---
    wlinesM = params['wlinesM'];
    pxlinesD0_base = params['pxlinesD0_base']
    pixdwavs_val = params['pixdwavs_val'];
    wavstep1 = params['wavstep1']
    calwav1 = params['calwav1'];
    dltFibY = params['dltFibY']

    # paramsからスキップリストを取得 (キーが存在しない場合はデフォルトの空リスト[])
    fibers_to_skip = params.get('fibers_to_skip', [])

    if fibers_to_skip:
        original_count = len(iFibAct)
        # np.setdiff1d を使い、iFibAct(処理対象) から fibers_to_skip(除外対象) を
        # 引いた差集合（含まれていないもの）を、新しい iFibAct にする
        iFibAct = np.setdiff1d(iFibAct, fibers_to_skip)

        print(f"  -> NOTE: Skipping {original_count - len(iFibAct)} fibers as requested: {fibers_to_skip}")
        print(f"  -> Now processing {len(iFibAct)} fibers.")

    pxlinesD0 = pxlinesD0_base + pixdwavs_val
    xpix = np.arange(nx)

    # --- 参照スペクトルの準備 ---
    if calwav1 >= 586 and calwav1 <= 596:
        fileCal = 'psgrad586-596.txt'
    else:
        print(f"  -> ERROR: No reference solar spectrum for wavelength range around {calwav1} nm.");
        sys.exit()
    try:
        print(f"  -> Loading solar spectrum: {fileCal}")
        spMdl = np.loadtxt(os.path.join(solar_spec_dir, fileCal), skiprows=14)
    except FileNotFoundError:
        print(f"  -> ERROR: Reference spectrum not found at {os.path.join(solar_spec_dir, fileCal)}");
        return

    for iJ in range(ny): spDat[iJ, :] = median_filter(spDat[iJ, :], size=3)

    # --- 参照スペクトルから精密な輝線波長を決定 ---
    print("  -> Refining line wavelengths from solar model...")
    wavs = np.zeros_like(wlinesM);
    #wavair2vac = 1.000000
    wavair2vac = 1.000276
    kernel = gaussian_kernel(size=181, sigma=61);
    xm = spMdl[:, 0] / wavair2vac
    cv = convolve(spMdl[:, 1], kernel, mode='same');
    ym = cv / np.median(cv)
    for idx, w_approx in enumerate(wlinesM):
        # 1. 粗いフィット
        xrng_wav1 = w_approx + np.array([-0.15, 0.15])  # 0.3 nm幅で探す
        ixm1 = (xm >= xrng_wav1[0]) & (xm <= xrng_wav1[1])
        if ixm1.sum() < 5: continue
        Aini1 = [np.min(ym[ixm1]) - 1, w_approx, 0.002, 1.0]
        res1 = least_squares(residuals4, Aini1, args=(xm[ixm1], ym[ixm1]), loss='huber')
        center_approx = res1.x[1]

        # 2. 精密なフィット
        xrng_wav2 = center_approx + np.array([-0.035, 0.035])  # 0.07 nm幅でさらに詰める
        ixm2 = (xm >= xrng_wav2[0]) & (xm <= xrng_wav2[1])
        if ixm2.sum() < 5: continue
        Aini2 = [np.min(ym[ixm2]) - 1, w_approx, 0.002, 1.0]  # 初期値も更新
        res2 = least_squares(residuals4, Aini2, args=(xm[ixm2], ym[ixm2]), loss='huber')

        wavs[idx] = res2.x[1]
    print(f"  -> Refined reference wavelengths [nm]: {np.round(wavs, 4)}")
    if flgPause: input("Press Enter to continue to fiber fitting...")

    # --- ファイバー毎の波長校正 ---
    wmp = np.zeros_like(spDat, dtype=np.float64)
    dltWMP = np.zeros((ny, len(wlinesM)));
    nBad = np.zeros(ny, dtype=int)

    y0even = pxlinesD0[0]
    y0odd = y0even + dltFibY

    fig_detail, axes_detail = plt.subplots(4, 4, figsize=(12, 10), dpi=96)
    plt.subplots_adjust(wspace=0.5, hspace=0.8, left=0.08, right=0.95, top=0.9, bottom=0.05)
    axes_detail_flat = axes_detail.flatten()

    for j in iFibAct:
        print(f"  -> Processing Fiber {j:03d}...")
        for ax in axes_detail_flat: ax.clear()
        iFig = 0

        if j % 2 == 0:
            pxlinesD_current = pxlinesD0 - pxlinesD0[0] + y0even
        else:
            pxlinesD_current = pxlinesD0 - pxlinesD0[0] + y0odd

        wpix_fit = np.full_like(pxlinesD_current, np.nan, dtype=np.float64)

        for idx, p_approx in enumerate(pxlinesD_current):
            try:
                y = spDat[j, :]
                xrng1 = p_approx + np.array([-0.5, 0.5]) * 0.12 / wavstep1
                ixd1 = (xpix >= max(0, xrng1[0])) & (xpix < min(nx, xrng1[1]))
                if ixd1.sum() < 5: continue
                max_val1 = np.max(y[ixd1]);
                if max_val1 == 0: continue
                Aini1 = [-1.0, np.mean(xpix[ixd1]), 5.0, 1.0]
                bounds1 = ([-1.5, np.min(xpix[ixd1]), 3.0, 0.0], [0.0, np.max(xpix[ixd1]), 20.0, 1.5])
                res1 = least_squares(residuals4, Aini1, args=(xpix[ixd1], y[ixd1] / max_val1), bounds=bounds1, loss='huber')
                center_approx = res1.x[1]

                xrng2 = center_approx + np.array([-0.5, 0.5]) * 0.07 / wavstep1
                ixd2 = (xpix >= max(0, xrng2[0])) & (xpix < min(nx, xrng2[1]))
                if ixd2.sum() < 5: continue
                max_val2 = np.max(y[ixd2]);
                if max_val2 == 0: continue

                #safe_center_guess = np.mean(xpix[ixd2])
                Aini2 = [-1, center_approx, 5, 1]
                #bounds2 = ([-1.5, np.min(xpix[ixd2]), 5, 0.1], [-0.1, np.max(xpix[ixd2]), 20, 1.5])
                bounds2 = ([-1.5, np.min(xpix[ixd2]), 2, 0.1], [-0.1, np.max(xpix[ixd2]), 20, 1.5])

                res2 = least_squares(residuals4, Aini2, args=(xpix[ixd2], y[ixd2] / max_val2), bounds=bounds2,
                                     loss='huber')
                par = res2.x
                wpix_fit[idx] = par[1]

                if iFig < len(axes_detail_flat):
                    ax = axes_detail_flat[iFig]
                    ax.plot(xpix[ixd1], y[ixd1], '.-', label='Data')
                    fit_y = gauss1d4(xpix[ixd2], par) * max_val2
                    ax.plot(xpix[ixd2], fit_y, '--', label='Fit')
                    ax.axvline(par[1], color='r', linestyle='--', linewidth=1)
                    ax.set_title(f"Line {idx}: {p_approx:.1f} -> {par[1]:.1f}", fontsize=7)
                    ax.set_xlim(xrng1);
                    ax.tick_params(labelsize=6)
                    if idx == 0: ax.legend(fontsize='xx-small')
                    iFig += 1

            except ValueError as e:
                print(f"    -> WARNING: Failed to fit Line {idx} for Fiber {j}. Skipping. Error: {e}")
                continue

        valid_indices = ~np.isnan(wpix_fit)
        ndeg = 5
        #ndeg = 4
        #ndeg = 1
        if np.sum(valid_indices) < ndeg + 1:
            print(f"  -> WARNING: Not enough valid points to fit for Fiber {j}. Skipping.")
            wmp[j, :] = np.nan
            continue

        """
        PROBLEM_FIBER_NUMBER = 103  # ← ここをエラーが出たファイバー番号に書き換える
        if j == PROBLEM_FIBER_NUMBER:
            plt.figure(figsize=(8, 6))
            plt.plot(wpix_fit[valid_indices], wavs[valid_indices], 'o-', label=f"Fiber {j} Fit Points")
            # どの点がどの波長か分かりやすくする
            for i, txt in enumerate(wavs[valid_indices]):
                plt.annotate(f"{txt:.3f}", (wpix_fit[valid_indices][i], wavs[valid_indices][i]))
            plt.xlabel("Fitted Pixel Position")
            plt.ylabel("Reference Wavelength (nm)")
            plt.title(f"Wavelength vs. Pixel Fit for Fiber {j}")
            plt.grid(True)
            plt.legend()
            plt.show()
        """
        coef = np.polyfit(wpix_fit[valid_indices], wavs[valid_indices], ndeg)
        pfit = np.poly1d(coef)
        residuals_nm = wavs - pfit(wpix_fit)
        dltWMP[j, :] = residuals_nm
        mask = np.abs(residuals_nm * 1000) <= 3.0
        if not np.all(mask[valid_indices]):
            good_fit_pts = valid_indices & mask
            if np.sum(good_fit_pts) >= ndeg + 1:
                coef = np.polyfit(wpix_fit[good_fit_pts], wavs[good_fit_pts], ndeg)
                pfit = np.poly1d(coef)
                dltWMP[j, :] = wavs - pfit(wpix_fit)
        nBad[j] = np.sum(~mask & valid_indices)
        wmp[j, :] = pfit(xpix)

        # ★★重要: 元のコードのロジックを完全に復元★★
        # 次のファイバーの予想位置を更新する
        ifunc = interp1d(wmp[j, :], xpix, kind='linear', fill_value='extrapolate', bounds_error=False)
        xpix0 = ifunc(wavs[0])
        if np.isnan(xpix0):
            print(f"  -> WARNING: Failed to calculate next starting position for Fiber {j}. Tracking may be inexact.")
            if j % 2 == 0:
                y0odd = y0even + dltFibY
            else:
                y0even = y0odd - dltFibY
        else:
            # 「リープフロッグ」方式で偶数・奇数両方のトラッカーを更新する
            if j % 2 == 0:
                y0even = xpix0  # 偶数ファイバーのトラッカーを更新
                y0odd = y0even + dltFibY  # 更新された偶数位置に基づき、次の奇数位置を予測
            else:  # j is odd
                y0odd = xpix0  # 奇数ファイバーのトラッカーを更新
                y0even = y0odd - dltFibY  # 更新された奇数位置に基づき、次の偶数位置を予測

        # プロットの保存と表示
        fig_detail.suptitle(f"Fiber {j} Fit Details (Bad Fits: {nBad[j]})", fontsize=16)
        if save_fiber_plots:
            png_filename = f"fiber_{j:03d}_details.png"
            fig_detail.savefig(os.path.join(plot_output_dir, png_filename))
        if flgPause:
            plt.draw();
            plt.pause(0.01)
            input(f"Finished processing Fiber {j}. Press Enter to continue...")

    print("  -> All fibers processed.")
    plt.close(fig_detail)

    # --- 結果の保存と最終プロット ---
    print(f"  -> Saving wavelength map to: {fileWMP}")
    hd_out = hd.copy()
    hd_out['HISTORY'] = 'Wavelength calibrated with mkWavMap4b_final.py'
    hd_out['BUNIT'] = ('nm', 'Wavelength unit')
    hdu_primary = fits.PrimaryHDU(data=wmp.astype(np.float32), header=hd_out)
    hdul_out = fits.HDUList([hdu_primary])
    if 'FIBERS' in fits.open(input_fsp_path): hdul_out.append(fits.open(input_fsp_path)['FIBERS'])
    if 'IFIBERS' in fits.open(input_fsp_path): hdul_out.append(fits.open(input_fsp_path)['IFIBERS'])
    os.makedirs(output_dir, exist_ok=True)
    hdul_out.writeto(fileWMP, overwrite=True)
    print("  -> Save complete.")

    print("  -> Displaying final summary plot. Close the window to exit.")
    fig_summary, axes_summary = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    fig_summary.suptitle(f'Wavelength Calibration Summary: {base_filename}', y=0.98)
    ax = axes_summary[0]
    median_wav = np.nanmedian(wmp)
    im = ax.imshow(wmp, cmap='jet', origin='lower', aspect='auto', interpolation='none', vmin=median_wav - 2,
                   vmax=median_wav + 2)
    ax.set_title('Wavelength Map (nm)');
    ax.set_xlabel('Dispersion Axis (pixels)');
    ax.set_ylabel('Spatial Axis (Fiber #)')
    fig_summary.colorbar(im, ax=ax)
    ax = axes_summary[1]
    std_pm = np.nanstd(dltWMP) * 1e3
    im = ax.imshow(dltWMP * 1e3, cmap='RdBu_r', origin='lower', aspect='auto', interpolation='none', vmin=-3, vmax=3)
    ax.set_title(f'Fit Residuals (pm) | StdDev={std_pm:.2f} pm')
    ax.set_xlabel('Reference Line Index');
    ax.set_ylabel('Spatial Axis (Fiber #)')
    fig_summary.colorbar(im, ax=ax, label='Wavelength Delta (pm)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ==============================================================================
# スクリプトの実行
# ==============================================================================
if __name__ == "__main__":
    base_dir = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/"
    master_sky_filepath = os.path.join(base_dir, "output/20251021/master_sky.fits")#変えるのはここ
    solar_spectrum_directory = os.path.join(base_dir, "psg/")
    calibration_params = {
        #'wlinesM': np.array([588.544, 589.157, 589.449, 589.756, 590.732, 591.580, 591.789]),
        #'wlinesM' : np.array([588.39, 588.995, 589.3, 589.592, 590.560, 591.002, 591.417]),
        'wlinesM': np.array([588.39, 588.995, 589.3, 589.592, 590.560,591.002, 591.417]),#7波長
        #'wlinesM': np.array([588.39, 588.995, 589.3, 589.592,  591.002, 591.417]),  # 6波長1
        #'wlinesM': np.array([588.39, 588.995, 589.3, 589.592, 590.560, 591.002, 591.417]),#6波長2
        #'wlinesM': np.array([588.39, 588.995, 589.592, 590.560, 591.002, 591.417]),  #202508
        #'wlinesM': np.array([588.995, 589.592,]),#2波長
        #'pxlinesD0_base': np.array([769, 949, 1035, 1127, 1423, 1557, 1685]),#202505
        #'pxlinesD0_base': np.array([852, 1034, 1119, 1220, 1507, 1637 ,1769]),#202506
        #'pxlinesD0_base': np.array([1012, 1198, 1289, 1380, 1811,1942,  2006]),#202507
        #'pxlinesD0_base': np.array([1012, 1198, 1380, 1811, 1942, 2006]),  #202508
        'pxlinesD0_base': np.array([1012, 1192, 1280, 1374, 1712, 1942, 2003]), #202508
        #'pxlinesD0_base': np.array([1003, 1195, 1281, 1373, 1712, 1942, 2003]),
        #'pxlinesD0_base': np.array([1012, 1192, 1280, 1374, 1942, 2003]),  # 202508 2
        #'pxlinesD0_base': np.array([790, 1139]),#20150223
        'pixdwavs_val': 0.0,
        'wavstep1': 0.00293,
        #'wavstep1': 0.00293080868,
        'calwav1': 588.9,
        #'dltFibY': -12.5,#基本はこっち
        'dltFibY': +12.5,#ファイバ番号０がうまくいかない時こっち

    # 'fibers_to_skip': []  # スキップしない場合
    'fibers_to_skip': [0],  # ファイバー0だけをスキップする場合
    # 'fibers_to_skip': [0, 10, 103], # 0, 10, 103番をスキップする場合

    }

    if not os.path.exists(master_sky_filepath):
        print(f"ERROR: Input file not found '{master_sky_filepath}'");
        sys.exit()

    mkWavMap4b_final(
        master_sky_filepath, solar_spectrum_directory, calibration_params,
        flgPause=False, save_fiber_plots=True
    )
    print("\n" + "=" * 80);
    print("All processing finished.");
    print("=" * 80)