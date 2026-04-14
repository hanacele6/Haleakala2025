import numpy as np
from astropy.io import fits
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ==============================================================================
# 2次元スペクトルマップの生成
# ==============================================================================
def generate_solar_subtracted_map(wc_fits_path, solar_model_norm, wl_model, fit_config, output_path):
    """
    Step 10で決定された太陽モデル(solar_model_norm)を使って、
    2次元スペクトル全体から太陽光を差し引き、.sub.fits を生成する。
    """
    if not wc_fits_path.exists():
        print(f"    -> WARNING: 2D FITS file not found for mapping: {wc_fits_path}")
        return

    try:
        with fits.open(wc_fits_path) as hdul:
            data_2d = hdul[0].data  # (N_fibers, N_wavelengths)
            header = hdul[0].header

            # WCSから波長軸を再構築
            nx = header['NAXIS1']
            crval1 = header['CRVAL1']
            cdelt1 = header['CDELT1']
            crpix1 = header.get('CRPIX1', 1.0)
            wl_2d = crval1 + (np.arange(nx) - (crpix1 - 1)) * cdelt1

        n_fibers = data_2d.shape[0]
        data_sub = np.zeros((n_fibers, len(wl_model)), dtype=np.float32)

        # モデルのフィッティング用行列 (A matrix for Ax = b)
        # [Model, Constant]
        A = np.vstack([solar_model_norm, np.ones(len(solar_model_norm))]).T

        # D2除外設定がある場合はマスクを作成
        use_mask = np.ones(len(wl_model), dtype=bool)
        if fit_config.get('exclude_d2_region', False):
            d2_wl = fit_config.get('d2_wavelength_nm', 589.0)
            hw_pix = fit_config.get('d2_exclusion_half_width_pix', 10)
            # モデル波長軸上でインデックスを探す
            idx_d2 = np.argmin(np.abs(wl_model - d2_wl))
            s = max(0, idx_d2 - hw_pix)
            e = min(len(wl_model), idx_d2 + hw_pix)
            use_mask[s:e] = False

        # --- 各ファイバーごとのループ処理 ---
        # 線形最小二乗法で Scale と Offset を決める
        for i in range(n_fibers):
            spec_obs = data_2d[i, :]

            # 観測データをモデルの波長グリッドに合わせる (線形補間)
            spec_resampled = np.interp(wl_model, wl_2d, spec_obs)

            # フィッティング (マスク適用)
            # lstsq は (Scale, Offset) を返す
            if np.sum(use_mask) > 5:
                res = np.linalg.lstsq(A[use_mask], spec_resampled[use_mask], rcond=None)
                coeffs = res[0]
                scale, offset = coeffs[0], coeffs[1]
            else:
                scale, offset = 0.0, 0.0

            # 引き算実行 (全領域)
            # Residual = Observed - (Scale * SolarModel + Offset)
            data_sub[i, :] = spec_resampled - (scale * solar_model_norm + offset)

        # FITS保存
        # ヘッダーを更新して保存（波長軸がクロップ/リサンプルされているため）
        new_header = header.copy()
        new_header['CRVAL1'] = wl_model[0]
        new_header['CDELT1'] = wl_model[1] - wl_model[0]
        new_header['CRPIX1'] = 1.0
        new_header['NAXIS1'] = len(wl_model)
        new_header['HISTORY'] = 'Solar subtracted using optimized model from Step 10'

        hdu = fits.PrimaryHDU(data=data_sub, header=new_header)
        hdu.writeto(output_path, overwrite=True)
        print(f"    -> Generated 2D subtracted map: {output_path.name}")

    except Exception as e:
        print(f"    -> ERROR generating 2D map: {e}")


# ==============================================================================
# メインの太陽光減算・絶対輝度変換関数
# ==============================================================================
def process_spectrum_original_logic(input_dat_path, solar_spec_path, hapke_path, output_dir,
                                    constants, fit_config):
    """
    d2固定だがデータ幅は可変。観測データを太陽光の波長範囲にクロップし、
    d2領域の除外をON/OFFできるバージョン。
    さらに、最適化されたモデルを使って2Dマップ(.sub.fits)を生成する。
    """
    sft_val = constants['sft']
    sft_suffix = f"_sft{int(sft_val * 10000):03d}"
    base_filename = f"{input_dat_path.stem}{sft_suffix}"

    print(f"\n  -> Processing: {base_filename} (sft={sft_val})")

    # --- 1. 入力ファイルの読み込み ---
    try:
        obs_data = np.loadtxt(input_dat_path, skiprows=1)
        wl, Nat = obs_data[:, 0], obs_data[:, 1]
        sol_data = np.loadtxt(solar_spec_path)
    except FileNotFoundError as e:
        print(f"    -> ERROR: Input file not found: {e}. Skipping.")
        return
    except Exception as e:
        print(f"    -> ERROR: Failed to read input files: {e}. Skipping.")
        return

    # --- 2. 物理定数と太陽光データの準備 ---
    Vme, Vms, Rmn, sft, Rmc, c = (constants['Vme'], constants['Vms'], constants['Rmn'],
                                  constants['sft'], constants['Rmc'], constants['c'])
    Shap = (Rmc / Rmn) ** 2 / 1e+4
    wavair_factor = 1.000
    sol_data[:, 0] = sol_data[:, 0] / wavair_factor

    direct_solar_wl = sol_data[:, 0] - sft
    reflected_solar_wl = direct_solar_wl * (1 + Vms / c) * (1 + Vme / c)

    # --- 3. 処理範囲の決定と観測データのクロップ ---
    valid_wl_min = max(np.min(direct_solar_wl), np.min(reflected_solar_wl))
    valid_wl_max = min(np.max(direct_solar_wl), np.max(reflected_solar_wl))

    original_count = len(wl)
    crop_mask = (wl >= valid_wl_min) & (wl <= valid_wl_max)
    wl, Nat = wl[crop_mask], Nat[crop_mask]

    if len(wl) < 20:
        print("    -> ERROR: No/few overlapping wavelength points found.")
        return

    ixm = len(wl)
    dwl = np.median(np.diff(wl))

    # --- 4. 補間用の太陽光モデルの最終準備 ---
    sol = np.zeros((sol_data.shape[0], 3))
    sol[:, 0] = direct_solar_wl
    sol[:, 1] = sol_data[:, 2]
    sol[:, 2] = sol_data[:, 1]
    wlsurf = reflected_solar_wl

    # --- 5. 線形補間 ---
    # (既存のロジック維持)
    iwm2, iws1, iws2 = sol.shape[0], 0, 0
    surf = np.zeros((ixm, 2), dtype=np.float64)
    for ix in range(ixm):
        for iw in range(iws1, iwm2 - 1):
            if (wl[ix] - sol[iw, 0]) * (wl[ix] - sol[iw + 1, 0]) <= 0:
                x1, x2, y1, y2 = sol[iw, 0], sol[iw + 1, 0], sol[iw, 1], sol[iw + 1, 1]
                surf[ix, 0] = y1 if (x2 - x1) == 0 else (y1 * (x2 - wl[ix]) + y2 * (wl[ix] - x1)) / (x2 - x1)
                iws1 = iw
                break
        for iw in range(iws2, iwm2 - 1):
            if (wl[ix] - wlsurf[iw]) * (wl[ix] - wlsurf[iw + 1]) <= 0:
                x1, x2, y1, y2 = wlsurf[iw], wlsurf[iw + 1], sol[iw, 2], sol[iw + 1, 2]
                surf[ix, 1] = y1 if (x2 - x1) == 0 else (y1 * (x2 - wl[ix]) + y2 * (wl[ix] - x1)) / (x2 - x1)
                iws2 = iw
                break

    # --- 6. 最適化ループの準備 (d2除外のON/OFF) ---
    if fit_config['exclude_d2_region']:
        d2_wl = fit_config['d2_wavelength_nm']
        half_width = fit_config['d2_exclusion_half_width_pix']
        d2_idx = np.argmin(np.abs(wl - d2_wl))
        exclude_start = max(0, d2_idx - half_width)
        exclude_end = min(ixm, d2_idx + half_width + 1)

        fit_mask = np.ones(ixm, dtype=bool)
        fit_mask[exclude_start:exclude_end] = False
        Nat2 = Nat[fit_mask]
        pix_range_fit = np.arange(ixm)[fit_mask]
        surf_slicer = lambda s: s[fit_mask]
    else:
        Nat2 = Nat
        pix_range_fit = np.arange(ixm)
        surf_slicer = lambda s: s

    if len(Nat2) < 3:
        return

    # --- 7. 最適化ループ ---
    zansamin = 1.0e+32
    best_params = {}

    for iFWHM in range(30, 101):
        FWHM = iFWHM * 0.1
        sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        psf = np.exp(-((np.arange(ixm, dtype=np.float64) - float(ixm) / 2.0) / sigma) ** 2 / 2.0)
        psf2 = psf / np.sum(psf)
        fft_surf0, fft_surf1 = np.fft.fft(surf[:, 0]), np.fft.fft(surf[:, 1])
        fft_psf2 = np.fft.fft(psf2)
        shift_amount = -int(ixm / 2)
        conv_surf0 = np.roll(np.real(np.fft.ifft(fft_surf0 * fft_psf2)), shift=shift_amount)
        conv_surf1 = np.roll(np.real(np.fft.ifft(fft_surf1 * fft_psf2)), shift=shift_amount)
        surf1 = np.column_stack([conv_surf0, conv_surf1])

        for iairm in range(51):
            airm = 0.1 * iairm
            surf1_clipped = np.clip(surf1, a_min=0, a_max=None)
            surf2 = surf1_clipped[:, 0] ** airm * surf1_clipped[:, 1]
            surf3 = surf_slicer(surf2)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratioa = Nat2 / surf3
            if not np.all(np.isfinite(ratioa)): continue

            aa0 = np.polyfit(pix_range_fit, ratioa, 2)[::-1]
            ratiof = aa0[0] + aa0[1] * pix_range_fit + aa0[2] * pix_range_fit ** 2
            zansa = np.sum((Nat2 - surf3 * ratiof) ** 2)

            if zansa <= zansamin:
                zansamin = zansa
                best_params = {
                    'ratiof_full': aa0[0] + aa0[1] * np.arange(ixm) + aa0[2] * np.arange(ixm) ** 2,
                    'airms': airm, 'FWHMs': FWHM, 'surf2s': surf2
                }

    if not best_params:
        print("    -> ERROR: Could not find any valid fit parameters.")
        return

    print(f"    -> Best fit: airm={best_params['airms']:.2f}, FWHM={best_params['FWHMs']:.2f}")

    # --- 8. 最終計算と保存 ---
    surf3s_final = best_params['surf2s']  # Full model without slicing

    if np.sum(surf3s_final ** 2) == 0:
        return

    # 全体積分の補正係数
    if fit_config['exclude_d2_region']:
        # マスクされた領域でratio2を計算
        Nat2_final = Nat[fit_mask]
        surf3s_masked = surf3s_final[fit_mask]
        ratio2 = np.sum(Nat2_final * surf3s_masked) / np.sum(surf3s_masked ** 2)
    else:
        ratio2 = np.sum(Nat * surf3s_final) / np.sum(surf3s_final ** 2)

    Natb = Nat - ratio2 * best_params['surf2s']

    # --- 1Dファイルの保存 ---
    output_test_path = output_dir / f"{base_filename}.test.dat"
    output_data1 = np.column_stack([wl, Nat, best_params['surf2s'] * best_params['ratiof_full']])
    np.savetxt(output_test_path, output_data1, fmt='%.8e', header="Wavelength Obs Model")

    try:
        hap = fits.getdata(hapke_path)
        tothap = np.sum(hap) * Shap * dwl * 1e+12
        cts2MR = tothap / ratio2

        calib_info_path = output_dir / "calibration_factors.csv"
        if not calib_info_path.exists():
            with open(calib_info_path, "w") as f: f.write("filename,cts2MR\n")
        with open(calib_info_path, "a") as f:
            f.write(f"{base_filename},{cts2MR:.8e}\n")

        output_exos_path = output_dir / f"{base_filename}.exos.dat"
        exos_data = np.column_stack([wl, Natb * cts2MR])
        np.savetxt(output_exos_path, exos_data, fmt='%.8e', header="Wavelength Flux(MR)")
        print(f"    -> Saved final 1D spectrum: {output_exos_path.name}")

        # ======================================================================
        # 2Dスペクトルマップ(.sub.fits)の生成
        # ======================================================================
        # 元の .wc.fits を探す
        # 名前規則: filename.totfib.dat -> filename.wc.fits
        orig_base = input_dat_path.name.replace(".totfib.dat", "").replace(".totfib_orig.dat", "")
        # 親ディレクトリの構成から推測
        wc_fits_path = input_dat_path.parent.parent / "1_fits" / f"{orig_base}.wc.fits"
        if not wc_fits_path.exists():
            wc_fits_path = input_dat_path.parent / f"{orig_base}.wc.fits"

        if wc_fits_path.exists():
            # ベストフィットした太陽モデル(形状)を使用
            # ratio2やcts2MRは各ピクセルごとに異なりうるため、ここでは適用せず、
            # 「形状(surf2s)」だけを渡して、各ピクセルで明るさをフィットさせる
            sub_fits_name = f"{base_filename}.sub.fits"
            output_sub_path = output_dir / sub_fits_name

            generate_solar_subtracted_map(
                wc_fits_path=wc_fits_path,
                solar_model_norm=best_params['surf2s'],  # 畳み込み済みの形状
                wl_model=wl,  # モデルの波長軸
                fit_config=fit_config,
                output_path=output_sub_path
            )
        else:
            print(f"    -> WARNING: Could not find source .wc.fits for 2D mapping.")

    except Exception as e:
        print(f"    -> ERROR in final conversion/saving: {e}")


# ==============================================================================
# パイプライン実行用モジュール
# ==============================================================================
def run(run_info, config):
    """
    パイプラインから呼び出される太陽光減算・絶対輝度変換の実行関数
    """
    output_dir = run_info["output_dir"]
    csv_file_path = run_info["csv_path"]
    date_str = run_info["date"]

    solar_conf = config.get("solar_subtraction", {})
    solar_spec_path = Path(solar_conf.get("solar_spec_path", ""))
    target_types = solar_conf.get("target_types", ['MERCURY'])
    fit_config = solar_conf.get("fit_config", {})
    sft_dict = solar_conf.get("sft_values", {})

    force_rerun = config.get("pipeline", {}).get("force_rerun_solar", False)

    #print(f"\n--- 太陽光モデル減算・絶対輝度(MR)変換 (Milillo Mode) を開始します ---")

    if not solar_spec_path.exists():
        print(f"エラー: 太陽スペクトルファイルが見つかりません: {solar_spec_path}")
        return

    try:
        df = pd.read_csv(csv_file_path)
        type_col = df.columns[1]
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return

    def find_file_in_folders(name, root, subdirs=None):
        p_root = root / name
        if p_root.exists(): return p_root
        if subdirs:
            for sd in subdirs:
                p_sub = root / sd / name
                if p_sub.exists(): return p_sub
        return None

    hapke_path = find_file_in_folders(f"Hapke{date_str}.fits", output_dir, ["1_fits"])

    for process_type in target_types:
        if type_col not in df.columns: continue
        target_df = df[df[type_col] == process_type]
        if target_df.empty: continue

        print(f"\n[{process_type}] {len(target_df)} 個のファイルを処理します...")

        for idx, (row_index, row) in enumerate(target_df.iterrows(), start=1):
            base_name = f"{process_type}{idx}_tr"

            input_file = find_file_in_folders(f"{base_name}.totfib.dat", output_dir, ["2_spectra"])
            if not input_file:
                input_file = find_file_in_folders(f"{base_name}.totfib_orig.dat", output_dir, ["2_spectra"])

            if not input_file:
                continue

            # sft_values の決定
            term_side = str(row.get('terminator_side', '')).strip().lower()
            if term_side == 'dawn':
                sft_values_to_test = sft_dict.get('dawn', [0.001])
            elif term_side == 'dusk':
                sft_values_to_test = sft_dict.get('dusk', [-0.0005])
            else:
                sft_values_to_test = sft_dict.get('default', [-0.001])

            try:
                constants_this_run = {
                    'Vme': row['mercury_earth_radial_velocity_km_s'],
                    'Vms': row['mercury_sun_radial_velocity_km_s'],
                    'Rmn': row['apparent_diameter_arcsec'],
                    'Rmc': 4.879e+8,
                    'c': 299792.458
                }
            except KeyError:
                continue

            for sft_val in sft_values_to_test:
                constants_this_run['sft'] = sft_val

                # スキップ判定 (1D結果ファイルで判定)
                sft_suffix = f"_sft{int(sft_val * 10000):03d}"
                output_exos_name = f"{input_file.stem}{sft_suffix}.exos.dat"
                is_processed = find_file_in_folders(output_exos_name, output_dir, ["2_spectra"])

                if is_processed and not force_rerun:
                    # print(f"    > 処理済みスキップ: {output_exos_name}")
                    continue

                process_spectrum_original_logic(
                    input_dat_path=input_file,
                    solar_spec_path=solar_spec_path,
                    hapke_path=hapke_path,
                    output_dir=output_dir,
                    constants=constants_this_run,
                    fit_config=fit_config
                )

    print("--- 太陽光モデル減算・絶対輝度変換・2Dマップ生成が完了しました ---")


if __name__ == "__main__":
    print("Use as module.")