import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import os
import pandas as pd


# ==============================================================================
# メインの波長校正・再サンプリング関数 (アルゴリズム変更なし)
# ==============================================================================
def mkWcalSpec_final(input_fsp_path, wavmap_path, wl_flat_path,
                     sky_flat_fsp_path=None, params=None,
                     save_plots=False, representative_fiber_plot=None,
                     apply_wl_flat=True, processing_range=None):
    """
    2Dスペクトルを波長校正し、等間隔の波長軸に再サンプリングします。
    """
    print("\n" + "=" * 80)
    print(f"Starting Wavelength Resampling for: {os.path.basename(input_fsp_path)}")

    # --- パラメータとデフォルト値の設定 ---
    if params is None:
        params = {}
    wavshift = params.get('wavshift', 0.0)
    interp_kind = params.get('interpolation_kind', 'quadratic')
    header_info = params.get('header_info', None)

    # --- 出力ファイル名の設定 ---
    base_filename = os.path.basename(input_fsp_path).replace(".fits", "")
    output_dir = os.path.dirname(input_fsp_path)
    # もし1_fitsの中から読み込んでいたら、出力は一つ上の親ディレクトリ（root）に出したい場合
    if os.path.basename(output_dir) == "1_fits":
        output_dir = os.path.dirname(output_dir)

    file_wc = os.path.join(output_dir, f"{base_filename}.wc.fits")
    # file_wc = os.path.join(output_dir, f"{base_filename}.wc_test.fits")
    file_dcb = os.path.join(output_dir, f"{base_filename}.dcb.fits")
    file_img = os.path.join(output_dir, f"{base_filename}.img.fits")
    plot_output_dir = os.path.join(output_dir, "wcal_plots", base_filename)

    if save_plots:
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"  -> Diagnostic plots will be saved to '{plot_output_dir}'")

    # PROCESS_WAV_MIN = 588.39
    # PROCESS_WAV_MAX = 591.417
    # print(f"  -> Processing with fixed wavelength range: {PROCESS_WAV_MIN:.2f} - {PROCESS_WAV_MAX:.2f} nm")

    # --- FITSファイルの読み込み ---
    try:
        # 1. ファイルを1つずつ読み込み、成功したかを確認する
        print(f"  -> Reading input spectrum: {os.path.basename(input_fsp_path)}")
        with fits.open(input_fsp_path) as hdul:
            spDat = hdul[0].data.astype(np.float64)
            hd = hdul[0].header

        print(f"  -> Reading wavelength map: {os.path.basename(wavmap_path)}")
        with fits.open(wavmap_path) as hdul:
            wmp = hdul[0].data.astype(np.float64)
            wavair_factor = 1.000276
            # wavair_factor = 1.000
            wmp = wmp * wavair_factor

        spFlt = None
        if apply_wl_flat and wl_flat_path and os.path.exists(wl_flat_path):
            print(f"  -> Reading white-light flat: {os.path.basename(wl_flat_path)}")
            with fits.open(wl_flat_path) as hdul:
                spFlt = hdul[0].data.astype(np.float64)

        spSky = None
        if sky_flat_fsp_path and os.path.exists(sky_flat_fsp_path):
            print(f"  -> Reading sky flat: {os.path.basename(sky_flat_fsp_path)}")
            with fits.open(sky_flat_fsp_path) as hdul:
                spSky = hdul[0].data.astype(np.float64)

        # 2. ヘッダー関連の情報を整理する
        ny, nx = spDat.shape
        if header_info:
            print("  -> Using hardcoded header parameters.")
            nFibX = header_info['NFIBX']
            nFibY = header_info['NFIBY']
            iFibAct = header_info['iFibAct']
            iFib = header_info['iFib']
        else:
            print("  -> Reading header info from FITS file.")
            nFibX, nFibY = hd.get('NFIBX', 1), hd.get('NFIBY', ny)
            with fits.open(input_fsp_path) as hdul:
                try:
                    iFibAct = hdul['IFIBERS'].data
                    iFib = hdul['FIBERS'].data
                except KeyError:
                    iFibAct = np.arange(ny)
                    iFib = np.arange(ny)

    except Exception as e:
        print("\n" + "!" * 80)
        print(f"  -> FATAL ERROR: FITSファイルの読み込み中に問題が発生しました。")
        print(f"  -> エラー内容: {e}")
        print("!" * 80 + "\n")
        return
    # --- 波長軸の再定義（リニアな等間隔グリッドを作成） ---
    wmp_shifted = wmp + wavshift
    


    if processing_range:
        print(
            f"  -> Defining new grid based on target range density: {processing_range[0]:.4f} - {processing_range[1]:.4f} nm")
        
        # 1. target_range内に元々ピクセルが何個あるか、全ファイバーで平均を取る
        pixels_in_range = []
        for j in iFibAct:
            wmp_j = wmp_shifted[j, :]
            # target_range内にあるピクセルの数を数える
            count = np.sum((wmp_j >= processing_range[0]) & (wmp_j <= processing_range[1]))
            if count > 0:
                pixels_in_range.append(count)

        if not pixels_in_range:
            raise ValueError("No valid pixels found within the target_range for any fiber.")

        # 平均ピクセル数を計算（小数点以下を丸めて整数に）
        avg_pixels_in_range = int(np.round(np.mean(pixels_in_range)))
        print(f"     Average original pixels in target range: {avg_pixels_in_range}")

        # 2. target_rangeの幅と平均ピクセル数から、新しい波長ステップを定義
        target_width = processing_range[1] - processing_range[0]
        # 新しいステップ (nm/pixel)
        rwstep_f = target_width / (avg_pixels_in_range - 1) if avg_pixels_in_range > 1 else target_width

        # 3. 新しい波長軸を構築する (元の総ピクセル数nxは維持する)
        # target_rangeが中心に来るように、全体の開始波長を決める
        rwmin_f = processing_range[0] - (rwstep_f * ((nx - avg_pixels_in_range) / 2))
        wavs = rwmin_f + rwstep_f * np.arange(nx, dtype=np.float64)
        # --- 新しいロジックここまで ---

    else:
        # processing_rangeが指定されていない場合は、従来通りの自動設定
        wav_min_end = np.nanmax(wmp_shifted[iFibAct, 0])
        wav_max_start = np.nanmin(wmp_shifted[iFibAct, nx - 1])
        if wav_min_end < wav_max_start:
            rwmin, rwmax_orig = wav_min_end, wav_max_start
        else:
            rwmin, rwmax_orig = wav_max_start, wav_min_end

        rwstep = np.abs((rwmax_orig - rwmin) / (nx - 1))
        rwmin_f = float(f"{rwmin:.8g}")
        rwstep_f = float(f"{rwstep:.4g}")
        wavs = rwmin_f + rwstep_f * np.arange(nx, dtype=np.float64)

    rwmax_f = wavs[-1]

    print(f"  -> Resampling to new linear wavelength axis:")
    print(f"     WAV_MIN : {rwmin_f:.4f} nm")
    print(f"     WAV_MAX : {rwmax_f:.4f} nm")
    print(f"     WAV_STEP: {rwstep_f:.5f} nm/pix")

    rwmid_f = (rwmin_f + rwmax_f) / 2

    print(f"  -> Resampling to new linear wavelength axis:")
    print(f"     WAV_MIN : {rwmin_f:.4f} nm")
    print(f"     WAV_MAX : {rwmax_f:.4f} nm")
    print(f"     WAV_STEP: {rwstep_f:.5f} nm/pix")

    # --- 各ファイバーの処理 ---
    spDatWC = np.zeros_like(spDat)
    spDatFlt = np.zeros_like(spDat)
    spDatImg = np.zeros((nFibY, nFibX))
    spDatDcb = np.zeros((nx, nFibY, nFibX))

    # スカイフラット関連の配列
    FibFF = np.ones(ny)
    spSkyWC = np.zeros_like(spDat) if spSky is not None else None
    spSkyImg = np.zeros((nFibY, nFibX)) if spSky is not None else None

    flat_mode = params.get('flat_mode', '2d_standard')
    flat_sigma = params.get('flat_smoothing_sigma', 50)

    print("  -> Processing and resampling each fiber...")

    master_1d_flat = None
    if apply_wl_flat and spFlt is not None and flat_mode == '1d_smoothed':
        print(f"    -> Mode '1d_smoothed' selected. Creating master 1D flat (sigma={flat_sigma})...")
        # 有効な全ファイバーのスペクトルの中央値を取り、ノイズに強い1Dプロファイルを作成
        raw_1d_flat = np.nanmedian(spFlt[iFibAct, :], axis=0)
        # ガウシアンフィルターで細かいピクセルムラや吸収線を平滑化
        master_1d_flat = gaussian_filter1d(raw_1d_flat, sigma=flat_sigma)


    # 1. ホワイトフラット補正と再サンプリング
    for j in iFibAct:
        # このファイバーの波長データが有効かチェック
        wmp_j = wmp_shifted[j, :]
        if np.all(np.isnan(wmp_j)):
            print(f"    -> WARNING: Fiber {j} has no valid wavelength data. Skipping.")
            continue  # このファイバーをスキップして次のループへ

            # 1. 波長マップがNaNだけ、または無限大を含む場合はスキップ
        if np.all(np.isnan(wmp_j)) or not np.all(np.isfinite(wmp_j)):
            print(f"    -> WARNING: Fiber {j} has invalid (NaN/Inf) wavelength data. Skipping.")
            continue

            # 2. 波長が単調増加または単調減少するかをチェック
        diffs = np.diff(wmp_j)
        is_monotonic_increasing = np.all(diffs > 0)
        is_monotonic_decreasing = np.all(diffs < 0)

        if not (is_monotonic_increasing or is_monotonic_decreasing):
            print(f"    -> WARNING: Fiber {j} is not monotonic. Skipping.")
            continue  # 条件を満たさない場合、このファイバーの処理を中断し、次のループへ

        spDat_to_resample = spDat[j, :].copy()  # 元データをコピー
        spSky_to_resample = spSky[j, :].copy() if spSky is not None else None

        # ホワイトフラット補正
        if apply_wl_flat and spFlt is not None:
            # モードによって使うフラットデータを切り替える
            if flat_mode == '1d_smoothed':
                flat_to_use = master_1d_flat
            else: # '2d_standard'
                flat_to_use = spFlt[j, :]

            # ゼロ除算回避
            valid_flat = flat_to_use > 1e-6 
            median_flat = np.nanmedian(flat_to_use[512:1536])

            # spDatFlt (プロット用) を計算
            spDatFlt[j, valid_flat] = (spDat[j, valid_flat] / flat_to_use[valid_flat]) * median_flat
            # 補間するデータも更新
            spDat_to_resample = spDatFlt[j, :]

            # スカイデータも同様に処理
            if spSky is not None:
                spSky_to_resample_temp = np.zeros_like(spSky[j, :])
                spSky_to_resample_temp[valid_flat] = (spSky[j, valid_flat] / flat_to_use[valid_flat]) * median_flat
                spSky_to_resample = spSky_to_resample_temp
        else:
            # フラット補正をしない場合
            spDatFlt[j, :] = spDat_to_resample

        # 波長軸に沿って再サンプリング
        ifunct = interp1d(wmp_j, spDat_to_resample, kind=interp_kind, fill_value="extrapolate",
                          bounds_error=False)
        spDatWC[j, :] = ifunct(wavs)
        if spSky is not None and spSky_to_resample is not None:
            ifunct_sky = interp1d(wmp_j, spSky_to_resample, kind=interp_kind, fill_value="extrapolate",
                                  bounds_error=False)
            spSkyWC[j, :] = ifunct_sky(wavs)

        # debug
        """
        diffs = np.diff(wmp_j)
        if np.any(diffs <= 0):
            print(f"!!! 問題発生: ファイバー {j} で波長が単調増加していません。")
            problem_indices = np.where(diffs <= 0)[0]
            for idx in problem_indices:
                print(f"  -> ピクセル {idx} と {idx + 1} の間で問題発生:")
                print(f"     wmp[{idx}] = {wmp_j[idx]}, wmp[{idx + 1}] = {wmp_j[idx + 1]}")


        # 波長軸に沿って再サンプリング (正しいデータを使用)
        ifunct = interp1d(wmp_j, spDat_to_resample, kind=interp_kind, fill_value="extrapolate",
                          bounds_error=False)
        spDatWC[j, :] = ifunct(wavs)

        # スカイデータも同様に処理
        if spSky is not None and spSky_to_resample is not None:
            ifunct_sky = interp1d(wmp_j, spSky_to_resample, kind=interp_kind, fill_value="extrapolate",
                                  bounds_error=False)
            spSkyWC[j, :] = ifunct_sky(wavs)
        """

    # 2. スカイフラット補正（ファイバー間の感度補正）
    if spSky is not None:
        print("  -> Applying sky flat correction...")
        temp_sky_median = np.zeros(ny)
        for j in iFibAct:
            # スキップされたファイバーなどでspSkyWCが0のままの可能性がある
            if np.any(spSkyWC[j, :]):
                temp_sky_median[j] = np.median(spSkyWC[j, 512:1536])

        median_of_medians = np.median(temp_sky_median[iFibAct])

        # FibFFを1で初期化（補正係数=1は「何もしない」という意味）
        FibFF = np.ones(ny)

        # スカイの明るさが0でない、有効なファイバーだけを対象に補正係数を計算
        valid_fibers = (temp_sky_median != 0)
        FibFF[valid_fibers] = median_of_medians / temp_sky_median[valid_fibers]

        # 計算した補正係数を適用
        for j in iFibAct:
            spDatWC[j, :] *= FibFF[j]

    # 3. 最終的な画像とデータキューブを作成
    for j in iFibAct:
        iy, ix = j // nFibX, j % nFibX
        spDatImg[iy, ix] = np.median(spDatWC[j, 512:1536])
        spDatDcb[:, iy, ix] = spDatWC[j, :]

    # --- FITSファイルの保存 ---
    def create_header(base_hd):
        hd_out = base_hd.copy()
        hd_out['HISTORY'] = 'Resampled to linear wavelength grid with mkWcalSpec_final.py'
        hd_out['INTERPK'] = (interp_kind, 'Interpolation kind for resampling')
        if spSky is not None:
            hd_out['SKYFLAT'] = 'Applied'
        else:
            hd_out['SKYFLAT'] = 'None'
        # WCSキーワードを追加
        hd_out['CTYPE1'] = 'WAVE'
        hd_out['CRPIX1'] = 1.0
        hd_out['CRVAL1'] = rwmin_f
        hd_out['CDELT1'] = rwstep_f
        hd_out['CUNIT1'] = 'nm'
        hd_out['CTYPE2'] = 'FIBERID'
        hd_out['CRPIX2'] = 1.0
        hd_out['CRVAL2'] = 0.0
        hd_out['CDELT2'] = 1.0 #もし2次元スペクトルを見たい場合はここを視野角に合わせた値に変更する必要がある。
        return hd_out

    # .wc.fits (波長校正済み2Dスペクトル)
    hd_wc = create_header(hd)
    fits.HDUList([
        fits.PrimaryHDU(data=spDatWC.astype(np.float32), header=hd_wc),
        fits.ImageHDU(data=iFib, name='FIBERS'),
        fits.ImageHDU(data=iFibAct, name='IFIBERS')
    ]).writeto(file_wc, overwrite=True)
    print(f"  -> Saved wavelength-calibrated spectra to: {os.path.basename(file_wc)}")

    # .dcb.fits (データキューブ)
    hd_dcb = create_header(hd)
    hd_dcb['NAXIS'] = 3
    hd_dcb['NAXIS1'] = nx
    hd_dcb['NAXIS2'] = nFibY
    hd_dcb['NAXIS3'] = nFibX
    # WCSも3次元用に更新
    hd_dcb['CTYPE3'] = 'FIBER_X'
    hd_dcb['CRPIX3'] = 1.0
    hd_dcb['CRVAL3'] = 0.0
    hd_dcb['CDELT3'] = 1.0 #もし2次元スペクトルを見たい場合はここを視野角に合わせた値に変更する必要がある。

    fits.HDUList([
        fits.PrimaryHDU(data=spDatDcb.astype(np.float32), header=hd_dcb),
        fits.ImageHDU(data=iFib, name='FIBERS'),
        fits.ImageHDU(data=iFibAct, name='IFIBERS')
    ]).writeto(file_dcb, overwrite=True)
    print(f"  -> Saved data cube to: {os.path.basename(file_dcb)}")

    # .img.fits (ファイバーバンドル再構成像)
    hd_img = create_header(hd)
    fits.HDUList([
        fits.PrimaryHDU(data=spDatImg.astype(np.float32), header=hd_img),
        fits.ImageHDU(data=iFib, name='FIBERS'),
        fits.ImageHDU(data=iFibAct, name='IFIBERS')
    ]).writeto(file_img, overwrite=True)
    print(f"  -> Saved reconstructed image to: {os.path.basename(file_img)}")

    # --- プロットの作成 ---
    if save_plots or representative_fiber_plot is not None:
        fibers_to_plot = iFibAct if save_plots and representative_fiber_plot is None else [representative_fiber_plot]
        for j in fibers_to_plot:
            if j not in iFibAct: continue
            iy, ix = j // nFibX, j % nFibX

            fig = plt.figure(figsize=(10, 12), dpi=100)
            gs = gridspec.GridSpec(5, 1, height_ratios=[2, 2, 2, 3, 3])

            # (1) 波長校正済みスペクトル
            ax1 = fig.add_subplot(gs[0])
            vrng = np.percentile(spDatWC[iFibAct, :], [1, 99])
            im = ax1.imshow(spDatWC, aspect='auto', origin='lower', vmin=vrng[0], vmax=vrng[1],
                            extent=[wavs[0], wavs[-1], 0, ny])
            ax1.set_title(f"Wavelength Calibrated Spectra ({os.path.basename(file_wc)})", fontsize=10)
            ax1.axhline(y=j, color='w', ls='--', lw=1)

            # (2) フラットフィールド済みスペクトル
            ax2 = fig.add_subplot(gs[1])
            vrng = np.percentile(spDatFlt[iFibAct, :], [1, 99])
            im = ax2.imshow(spDatFlt, aspect='auto', origin='lower', vmin=vrng[0], vmax=vrng[1])
            ax2.set_title(f"Flat-Fielded Spectra (before resampling)", fontsize=10)
            ax2.axhline(y=j, color='w', ls='--', lw=1)

            # (3) 1次元スペクトルプロット
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(wavs, spDatWC[j, :], label=f"Fiber {j} (Object)")
            if spSkyWC is not None:
                ax3.plot(wavs, spSkyWC[j, :], label="Sky", alpha=0.7)
            ax3.set_title(f"1D Spectrum for Fiber {j}", fontsize=10)
            ax3.set_xlabel("Wavelength (nm)")
            ax3.set_ylabel("Intensity")
            ax3.grid(True, ls=':', alpha=0.5)
            ax3.legend()

            # (4) & (5) 再構成像と感度補正係数
            gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3:])
            ax4 = fig.add_subplot(gs_bottom[0])
            vrng = np.percentile(spDatImg, [1, 99])
            im = ax4.imshow(spDatImg, origin='lower', vmin=vrng[0], vmax=vrng[1])
            ax4.set_title(f"Reconstructed Image ({os.path.basename(file_img)})", fontsize=10)
            rect = patches.Rectangle((ix - 0.5, iy - 0.5), 1, 1, lw=2, ec='r', fc='none')
            ax4.add_patch(rect)
            fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

            if spSky is not None:
                ax5 = fig.add_subplot(gs_bottom[1])
                ax5.plot(iFibAct, FibFF[iFibAct], '.-')
                ax5.set_title("Sky Flat Correction Factor", fontsize=10)
                ax5.set_xlabel("Fiber ID")
                ax5.set_ylabel("Correction Factor")
                ax5.grid(True, ls=':', alpha=0.5)

            fig.suptitle(f"Diagnostics for {base_filename} | Fiber {j}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            if save_plots:
                png_filename = os.path.join(plot_output_dir, f"wcal_diag_{j:03d}.png")
                fig.savefig(png_filename)
                print(f"    -> Saved plot: {os.path.basename(png_filename)}")

            if representative_fiber_plot is not None:
                plt.show()

            plt.close(fig)

    print("=" * 80)
    print("All processing finished.")
    print("=" * 80)

    pass


# ==============================================================================
# パイプライン実行用モジュール
# ==============================================================================
def run(run_info, config):
    """
    パイプラインから呼び出される波長再サンプリング・フラット補正の実行関数
    """
    output_dir = run_info["output_dir"]
    csv_file_path = run_info["csv_path"]

    # config.yaml から設定を読み込む
    resample_conf = config.get("resample", {})
    target_types = resample_conf.get("target_types", ['MERCURY'])
    master_sky_flat_name = resample_conf.get("master_sky_flat", "master_sky")
    apply_wl_flat = resample_conf.get("apply_wl_flat", True)
    save_plots = resample_conf.get("save_plots", False)

    params_conf = resample_conf.get("params", {})
    wavshift = params_conf.get("wavshift", 0.0)
    interp_kind = params_conf.get("interpolation_kind", 'quadratic')
    target_range = tuple(params_conf.get("target_range", [588.4993, 590.4566]))
    bad_fibers = params_conf.get("bad_fibers", [])
    n_fib_x = params_conf.get("n_fib_x", 10)
    n_fib_y = params_conf.get("n_fib_y", 12)

    # '2d_standard' (従来) または '1d_smoothed' (同期間での白色光のデータがないとき)
    flat_mode = params_conf.get("flat_mode", "2d_standard")
    flat_sigma = params_conf.get("flat_smoothing_sigma", 50)

    force_rerun = config.get("pipeline", {}).get("force_rerun_resample", False)

    print(f"\n--- 波長再サンプリング・フラット補正を開始します ---")

    # ファイバー情報の構築
    all_fibers = np.arange(n_fib_x * n_fib_y)
    good_fibers = np.setdiff1d(all_fibers, bad_fibers)
    header_info = {
        'NFIBX': n_fib_x,
        'NFIBY': n_fib_y,
        'iFibAct': good_fibers,
        'iFib': all_fibers
    }
    process_params = {
        'wavshift': wavshift,
        'interpolation_kind': interp_kind,
        'header_info': header_info,
        'flat_mode': flat_mode,              
        'flat_smoothing_sigma': flat_sigma   
    }

    try:
        df = pd.read_csv(csv_file_path)
        type_col = df.columns[1]
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return

    # ★★★ フォルダ探索用ヘルパー関数 ★★★
    def find_file_in_folders(name, root):
        p_root = root / name
        p_fits = root / "1_fits" / name
        if p_root.exists(): return p_root
        if p_fits.exists(): return p_fits
        return None

    # マスターファイルのパス構築 (探索機能付き)
    wavmap = find_file_in_folders(f"{master_sky_flat_name}.wmp.fits", output_dir)
    sky_flat = find_file_in_folders(f"{master_sky_flat_name}.fits", output_dir)

    # ★★★ フラット画像 (HLG or LED) を自動で探す ★★★
    wl_flat = None
    if apply_wl_flat:
        hlg_path = find_file_in_folders("master_hlg.fits", output_dir)
        led_path = find_file_in_folders("master_led.fits", output_dir)

        if hlg_path:
            wl_flat = hlg_path
            print(f"  > ホワイトフラットとして '{wl_flat.name}' を使用します。")
        elif led_path:
            wl_flat = led_path
            print(f"  > ホワイトフラットとして '{wl_flat.name}' を使用します。")
        else:
            print(f"  > 警告: ホワイトフラット (master_hlg.fits または master_led.fits) が見つかりません！")
            print(f"    処理を中断します。Step 05 で正しく合成されているか確認してください。")
            return

    for process_type in target_types:
        if type_col not in df.columns: continue
        target_df = df[df[type_col] == process_type]

        if target_df.empty:
            continue

        print(f"\n[{process_type}] {len(target_df)} 個のファイルを処理します...")

        for i, _ in enumerate(target_df.iterrows(), start=1):
            target_base_name = f"{process_type}{i}_tr"

            # 入力ファイルを探索
            input_fsp = find_file_in_folders(f"{target_base_name}.fits", output_dir)

            # 出力予定のファイルパス (スキップ判定用)
            file_wc = output_dir / f"{target_base_name}.wc.fits"
            file_wc_sub = output_dir / "1_fits" / f"{target_base_name}.wc.fits"

            # ▼▼▼ スキップ処理 (サブフォルダ内もチェック) ▼▼▼
            if (file_wc.exists() or file_wc_sub.exists()) and not force_rerun:
                print(f"  > 処理済みスキップ: {target_base_name}")
                continue

            print(f"  > 処理中: {target_base_name}")

            # 必要なファイルが揃っているかチェック
            if not input_fsp or not input_fsp.exists():
                print(f"    > 警告: 入力スペクトルが見つかりません ({target_base_name}.fits)")
                continue
            if not wavmap or not wavmap.exists():
                print(f"    > 警告: 波長マップが見つかりません ({master_sky_flat_name}.wmp.fits)")
                continue

            mkWcalSpec_final(
                input_fsp_path=str(input_fsp),
                wavmap_path=str(wavmap),
                wl_flat_path=str(wl_flat),
                sky_flat_fsp_path=str(sky_flat) if sky_flat and sky_flat.exists() else None,
                params=process_params,
                save_plots=save_plots,
                apply_wl_flat=apply_wl_flat,
                processing_range=target_range
            )

    print("--- 波長再サンプリング・フラット補正が完了しました ---")


if __name__ == "__main__":
    print("This script is a module.")