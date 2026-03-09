import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import convolve
from scipy.optimize import least_squares
from astropy.io import fits
import os
from pathlib import Path

plt.rcParams.update({'font.size': 8})
plt.rcParams['axes.unicode_minus'] = False


def gauss1d4(x, A):
    z = (x - A[1]) / A[2]
    return A[0] * np.exp(-(z ** 2) / 2) + A[3]


def residuals4(A, x, y):
    return gauss1d4(x, A) - y


def gaussian_kernel(size, sigma=1):
    x = np.linspace(-size // 2, size // 2, size)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def mkWavMap4b_final(input_fsp_path, solar_spec_dir, params, save_fiber_plots=True, root_output_dir=None):
    base_filename = input_fsp_path.name

    # 整理フォルダの中のファイルを読み込んでも、出力は必ず直下に出す
    output_dir = Path(root_output_dir) if root_output_dir else input_fsp_path.parent
    fileWMP = output_dir / base_filename.replace(".fits", ".wmp.fits")
    plot_output_dir = output_dir / "fiber_plots" / input_fsp_path.stem

    if save_fiber_plots:
        plot_output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(input_fsp_path) as hdul:
        spDat = hdul[0].data.astype(np.float64)
        hd = hdul[0].header
        ny, nx = spDat.shape
        iFibAct = hdul['IFIBERS'].data if 'IFIBERS' in hdul else np.arange(ny)

    wlinesM = np.array(params['wlinesM'])
    pxlinesD0_base = np.array(params['pxlinesD0_base'])
    pixdwavs_val = params['pixdwavs_val']
    wavstep1 = params['wavstep1']
    calwav1 = params['calwav1']
    dltFibY = params.get('dltFibY', 12.5)
    fibers_to_skip = params.get('fibers_to_skip', [])
    if fibers_to_skip: iFibAct = np.setdiff1d(iFibAct, fibers_to_skip)

    xpix = np.arange(nx)

    fileCal = 'psgrad586-596.txt'
    spMdl = np.loadtxt(os.path.join(solar_spec_dir, fileCal), skiprows=14)

    wavs = np.zeros_like(wlinesM)
    wavair2vac = 1.000276
    kernel = gaussian_kernel(size=181, sigma=61)
    xm = spMdl[:, 0] / wavair2vac
    cv = convolve(spMdl[:, 1], kernel, mode='same')
    ym = cv / np.median(cv)

    for idx, w_approx in enumerate(wlinesM):
        xrng_wav1 = w_approx + np.array([-0.15, 0.15])
        ixm1 = (xm >= xrng_wav1[0]) & (xm <= xrng_wav1[1])
        if ixm1.sum() < 5: continue
        res1 = least_squares(residuals4, [np.min(ym[ixm1]) - 1, w_approx, 0.002, 1.0], args=(xm[ixm1], ym[ixm1]),
                             loss='huber')
        wavs[idx] = res1.x[1]

    wmp = np.zeros_like(spDat, dtype=np.float64)
    dltWMP = np.zeros((ny, len(wlinesM)))
    nBad = np.zeros(ny, dtype=int)

    fig_detail, axes_detail = plt.subplots(4, 4, figsize=(12, 10), dpi=96)
    axes_detail_flat = axes_detail.flatten()

    # =====================================================================
    # 偶数奇数完全分離
    # =====================================================================
    even_fibs = [f for f in iFibAct if f % 2 == 0]
    odd_fibs = [f for f in iFibAct if f % 2 != 0]

    def create_center_out_lists(fibs_list):
        if not len(fibs_list): return [], [], []
        c_val = np.median(fibs_list)
        c_fib = fibs_list[np.argmin(np.abs(np.array(fibs_list) - c_val))]
        up_list = sorted([f for f in fibs_list if f > c_fib])
        down_list = sorted([f for f in fibs_list if f < c_fib], reverse=True)
        return [c_fib], up_list, down_list

    even_c, even_up, even_d = create_center_out_lists(even_fibs)
    odd_c, odd_up, odd_d = create_center_out_lists(odd_fibs)

    # 処理順: 偶数の中央→上→下、その後に奇数の中央→上→下
    process_order = even_c + even_up + even_d + odd_c + odd_up + odd_d

    # グループ別の記憶（バトン）
    valley_memory_even = {}
    valley_memory_odd = {}

    for j in process_order:
        for ax in axes_detail_flat: ax.clear()
        iFig = 0
        y_smooth = median_filter(spDat[j, :], size=5)
        is_even = (j % 2 == 0)

        # -------------------------------------------------------------
        # 1. 探索基準位置の決定
        # -------------------------------------------------------------
        expected_pos = None
        search_window = 15  # 基本は隣なので±15ピクセル

        if is_even:
            memory = valley_memory_even
            if j in even_c:
                # 偶数の最初の中央ファイバー（網を広く）
                expected_pos = pxlinesD0_base[0] + pixdwavs_val
                search_window = 80
            elif j in even_up:
                prev_j = j - 2
                while prev_j not in memory and prev_j >= min(even_fibs): prev_j -= 2
                if prev_j in memory: expected_pos = memory[prev_j]
            elif j in even_d:
                prev_j = j + 2
                while prev_j not in memory and prev_j <= max(even_fibs): prev_j += 2
                if prev_j in memory: expected_pos = memory[prev_j]
        else:
            memory = valley_memory_odd
            if j in odd_c:
                # すでに絶対確実な位置を見つけている「中央の偶数ファイバー」を基準にする
                if len(even_c) > 0 and even_c[0] in valley_memory_even:
                    expected_pos = valley_memory_even[even_c[0]]
                    # 探索範囲は「± dltFibY」がすっぽり入る幅（余裕を見て+5）
                    search_window = max(20, int(abs(dltFibY) + 5))
                else:
                    # 万が一偶数が全滅していた場合の予備
                    expected_pos = pxlinesD0_base[0] + pixdwavs_val + dltFibY
                    search_window = 80
            elif j in odd_up:
                prev_j = j - 2
                while prev_j not in memory and prev_j >= min(odd_fibs): prev_j -= 2
                if prev_j in memory: expected_pos = memory[prev_j]
            elif j in odd_d:
                prev_j = j + 2
                while prev_j not in memory and prev_j <= max(odd_fibs): prev_j += 2
                if prev_j in memory: expected_pos = memory[prev_j]

        # 万が一、隣の記憶が全滅していた場合のフォールバック
        if expected_pos is None:
            expected_pos = pxlinesD0_base[0] + pixdwavs_val + (0 if is_even else dltFibY)
            search_window = 80

        search_center = int(expected_pos)
        s_start = max(0, search_center - search_window)
        s_end = min(nx, search_center + search_window)
        if s_start >= s_end: s_start, s_end = 0, nx

        # -------------------------------------------------------------

        actual_local_valley = np.argmin(y_smooth[s_start:s_end]) + s_start
        pxlinesD_current = (pxlinesD0_base - pxlinesD0_base[0]) + actual_local_valley
        wpix_fit = np.full_like(pxlinesD_current, np.nan, dtype=np.float64)

        for idx, p_approx in enumerate(pxlinesD_current):
            try:
                y = spDat[j, :]
                xrng1 = p_approx + np.array([-0.5, 0.5]) * 0.12 / wavstep1
                ixd1 = (xpix >= max(0, xrng1[0])) & (xpix < min(nx, xrng1[1]))
                if ixd1.sum() < 5: continue
                max_val1 = np.max(y[ixd1])
                if max_val1 == 0: continue
                res1 = least_squares(residuals4, [-1.0, np.mean(xpix[ixd1]), 5.0, 1.0],
                                     args=(xpix[ixd1], y[ixd1] / max_val1),
                                     bounds=([-1.5, np.min(xpix[ixd1]), 3.0, 0.0],
                                             [0.0, np.max(xpix[ixd1]), 20.0, 1.5]), loss='huber')

                xrng2 = res1.x[1] + np.array([-0.5, 0.5]) * 0.07 / wavstep1
                ixd2 = (xpix >= max(0, xrng2[0])) & (xpix < min(nx, xrng2[1]))
                if ixd2.sum() < 5: continue
                max_val2 = np.max(y[ixd2])
                if max_val2 == 0: continue
                res2 = least_squares(residuals4, [-1, res1.x[1], 5, 1],
                                     args=(xpix[ixd2], y[ixd2] / max_val2),
                                     bounds=([-1.5, np.min(xpix[ixd2]), 2, 0.1], [-0.1, np.max(xpix[ixd2]), 20, 1.5]),
                                     loss='huber')
                wpix_fit[idx] = res2.x[1]

                if iFig < len(axes_detail_flat):
                    ax = axes_detail_flat[iFig]
                    ax.plot(xpix[ixd1], y[ixd1], '.-')
                    ax.plot(xpix[ixd2], gauss1d4(xpix[ixd2], res2.x) * max_val2, '--')
                    ax.axvline(res2.x[1], color='r', linestyle='--', linewidth=1)
                    ax.set_title(f"L{idx}: {p_approx:.1f}->{res2.x[1]:.1f}", fontsize=7)
                    iFig += 1
            except ValueError:
                continue

        valid_indices = ~np.isnan(wpix_fit)
        ndeg = 3

        if np.sum(valid_indices) >= ndeg + 1:
            coef = np.polyfit(wpix_fit[valid_indices], wavs[valid_indices], ndeg)
            pfit = np.poly1d(coef)
            residuals_nm = wavs - pfit(wpix_fit)
            dltWMP[j, :] = residuals_nm

            mask = np.abs(residuals_nm * 1000) <= 5.0
            if not np.all(mask[valid_indices]):
                good_fit_pts = valid_indices & mask
                if np.sum(good_fit_pts) >= ndeg + 1:
                    coef = np.polyfit(wpix_fit[good_fit_pts], wavs[good_fit_pts], ndeg)
                    pfit = np.poly1d(coef)
                    dltWMP[j, :] = wavs - pfit(wpix_fit)

            nBad[j] = np.sum(~mask & valid_indices)
            wmp[j, :] = pfit(xpix)

            # 成功したファイバーの位置をそれぞれの記憶に保存
            if np.sum(valid_indices) >= 4 and nBad[j] <= 2:
                if is_even:
                    valley_memory_even[j] = actual_local_valley
                else:
                    valley_memory_odd[j] = actual_local_valley
        else:
            wmp[j, :] = np.nan
            nBad[j] = 999

        if save_fiber_plots:
            fig_detail.suptitle(f"Fiber {j} Fit Details (Bad: {nBad[j]})", fontsize=16)
            fig_detail.savefig(plot_output_dir / f"fiber_{j:03d}_details.png")

    plt.close(fig_detail)

    valid_dlt = dltWMP[~np.isnan(dltWMP)]
    std_pm = np.std(valid_dlt) * 1e3 if len(valid_dlt) > 0 else 9999.9
    print(f"  > 較正精度 (残差の標準偏差): {std_pm:.2f} pm")

    hd_out = hd.copy()
    hd_out['HISTORY'] = 'Wavelength calibrated (Center-Out Tracking & Even/Odd Split)'
    fits.HDUList([fits.PrimaryHDU(data=wmp.astype(np.float32), header=hd_out)]).writeto(fileWMP, overwrite=True)
    print(f"  > 波長マップ保存完了: {fileWMP.name}")

    # =====================================================================
    # ★★★ サマリー画像のプロット保存機能  ★★★
    # =====================================================================
    fig_summary, axes_summary = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
    fig_summary.suptitle(f'Wavelength Calibration Summary: {base_filename}', y=0.98)

    median_wav = np.nanmedian(wmp)
    im1 = axes_summary[0].imshow(wmp, cmap='jet', origin='lower', aspect='auto', vmin=median_wav - 2,
                                 vmax=median_wav + 2)
    axes_summary[0].set_title('Wavelength Map (nm)')
    fig_summary.colorbar(im1, ax=axes_summary[0])

    im2 = axes_summary[1].imshow(dltWMP * 1e3, cmap='RdBu_r', origin='lower', aspect='auto', vmin=-5, vmax=5)
    axes_summary[1].set_title(f'Fit Residuals (pm) | StdDev={std_pm:.2f} pm')
    fig_summary.colorbar(im2, ax=axes_summary[1], label='Wavelength Delta (pm)')

    plt.tight_layout()
    summary_plot_path = output_dir / f"summary_{base_filename.replace('.fits', '.png')}"
    plt.savefig(summary_plot_path)
    plt.close(fig_summary)
    print(f"  > サマリー画像を保存しました: {summary_plot_path.name}")
    # =====================================================================


def run(run_info, config):
    output_dir = run_info["output_dir"]
    force_rerun = config.get("pipeline", {}).get("force_rerun_wavelength", False)

    print(f"\n--- 波長較正処理を開始します ---")

    # 出力ファイルがすでに整理フォルダ(1_fits)にあるかチェックしてスキップ
    wmp_name = "master_sky.wmp.fits"
    wmp_path_direct = output_dir / wmp_name
    wmp_path_organized = output_dir / "1_fits" / wmp_name

    if (wmp_path_direct.exists() or wmp_path_organized.exists()) and not force_rerun:
        print(f"  > 処理済みスキップ: {wmp_name}")
        print("--- 波長較正処理完了 ---")
        return

    # 入力ファイル(master_sky.fits)を直下と 1_fits/ の両方から探す
    input_path = output_dir / "master_sky.fits"
    if not input_path.exists():
        input_path = output_dir / "1_fits" / "master_sky.fits"

    if not input_path.exists():
        print(f"エラー: 波長較正の基準となる {input_path.name} が見つかりません。")
        return

    mkWavMap4b_final(
        input_fsp_path=input_path,
        solar_spec_dir=config.get("wavelength", {}).get("solar_spec_dir", ""),
        params=config.get("wavelength", {}).get("params", {}),
        save_fiber_plots=config.get("wavelength", {}).get("save_fiber_plots", True),
        root_output_dir=output_dir  # 直下に保存させるためのパス渡し
    )
    print("--- 波長較正処理完了 ---")


if __name__ == "__main__":
    print("This script is a module.")