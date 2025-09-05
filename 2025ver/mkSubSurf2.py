import numpy as np
from astropy.io import fits
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt


def process_spectrum_original_logic(input_dat_path, solar_spec_path, hapke_path, output_dir,
                                    constants, fit_config):
    """
    d2固定だがデータ幅は可変。観測データを太陽光の波長範囲にクロップし、
    d2領域の除外をON/OFFできるバージョン。
    フィッティング波長範囲の手動指定、および複数領域のフィット除外機能を追加。
    """
    sft_val = constants['sft']
    sft_suffix = f"_sft{int(sft_val * 10000):03d}"
    base_filename = f"{input_dat_path.stem}{sft_suffix}"

    print(f"\n  -> Processing: {base_filename} (sft={sft_val:.6f})")

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
    solar_range_min = max(np.min(direct_solar_wl), np.min(reflected_solar_wl))
    solar_range_max = min(np.max(direct_solar_wl), np.max(reflected_solar_wl))
    print(f"    -> Available solar model range: {solar_range_min:.4f} - {solar_range_max:.4f} nm")

    user_wl_range = fit_config.get('wavelength_range_nm')
    final_min, final_max = solar_range_min, solar_range_max

    if user_wl_range and len(user_wl_range) == 2:
        user_min, user_max = min(user_wl_range), max(user_wl_range)
        print(f"    -> User-specified fitting range: {user_min:.4f} - {user_max:.4f} nm")
        final_min = max(final_min, user_min)
        final_max = min(final_max, user_max)
        print(f"    -> Applying intersected range for fitting: {final_min:.4f} - {final_max:.4f} nm")
    else:
        print(f"    -> No user range specified. Using full available solar range for fitting.")

    original_count = len(wl)
    crop_mask = (wl >= final_min) & (wl <= final_max)
    wl, Nat = wl[crop_mask], Nat[crop_mask]

    if len(wl) < 20:
        print(f"    -> ERROR: No/few overlapping wavelength points ({len(wl)} found) in the final range [{final_min:.4f}, {final_max:.4f}]. Skipping.")
        return

    ixm = len(wl)
    dwl = np.median(np.diff(wl))
    print(f"    -> Cropped observation data to final range. Points: {original_count} -> {ixm}")


    # --- 4. 補間用の太陽光モデルの最終準備 ---
    sol = np.zeros((sol_data.shape[0], 3))
    sol[:, 0] = direct_solar_wl
    sol[:, 1] = sol_data[:, 2]
    sol[:, 2] = sol_data[:, 1]
    wlsurf = reflected_solar_wl



    # --- 5. 線形補間 ---
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

    # --- 6. 最適化ループの準備 (フィット除外領域の処理) ---
    fit_mask = np.ones(ixm, dtype=bool)
    print("    -> Preparing fit exclusion mask...")

    if fit_config.get('exclude_d2_region', False):
        d2_wl = fit_config['d2_wavelength_nm']
        half_width = fit_config['d2_exclusion_half_width_pix']
        d2_idx = np.argmin(np.abs(wl - d2_wl))
        print(f"    -> D2 line ({d2_wl} nm) found near pixel index: {d2_idx}")

        exclude_start = max(0, d2_idx - half_width)
        exclude_end = min(ixm, d2_idx + half_width + 1)
        print(f"    -> Applying D2 exclusion mask for pixel range: {exclude_start} to {exclude_end - 1}")
        fit_mask[exclude_start:exclude_end] = False
    else:
        print("    -> D2 region exclusion is OFF.")

    custom_exclusion_ranges = fit_config.get('exclusion_wavelength_ranges_nm')
    if custom_exclusion_ranges and isinstance(custom_exclusion_ranges, list):
        print("    -> Applying custom wavelength exclusion masks...")
        for i, wl_range in enumerate(custom_exclusion_ranges):
            if isinstance(wl_range, list) and len(wl_range) == 2:
                start_wl, end_wl = min(wl_range), max(wl_range)
                range_mask = (wl >= start_wl) & (wl <= end_wl)
                num_excluded = np.sum(range_mask)
                if num_excluded > 0:
                     fit_mask[range_mask] = False
                     print(f"       - Range {i+1}: Excluded {num_excluded} pixels between {start_wl:.4f} nm and {end_wl:.4f} nm.")
                else:
                     print(f"       - Range {i+1}: No pixels found in range {start_wl:.4f} nm to {end_wl:.4f} nm.")
            else:
                print(f"       - WARNING: Skipping invalid range format: {wl_range}")
    else:
        print("    -> No custom wavelength exclusion ranges specified.")

    if np.all(fit_mask):
        print("    -> Fitting will use the full spectrum (no exclusions applied).")
        Nat2 = Nat
        pix_range_fit = np.arange(ixm)
        surf_slicer = lambda s: s
    else:
        Nat2 = Nat[fit_mask]
        pix_range_fit = np.arange(ixm)[fit_mask]
        surf_slicer = lambda s: s[fit_mask]
        print(f"    -> Total points for fitting after applying all masks: {len(Nat2)} / {ixm}")

    if len(Nat2) < 3:
        print("    -> ERROR: Not enough data points to perform fit after exclusion. Skipping.")
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
        print("    -> ERROR: Could not find any valid fit parameters. Skipping.")
        return

    print(f"    -> Best fit: airm={best_params['airms']:.2f}, FWHM={best_params['FWHMs']:.2f}, zansa={zansamin:.4e}")


    # --- 8. 最終計算と保存 ---
    if not np.all(fit_mask):
        Nat2_final = Nat[fit_mask]
        surf3s_final = surf_slicer(best_params['surf2s'])
    else:
        Nat2_final = Nat
        surf3s_final = best_params['surf2s']

    if np.sum(surf3s_final ** 2) == 0:
        print("    -> ERROR: Cannot calculate ratio2, model sum is zero. Skipping final save.")
        return

    ratio2 = np.sum(Nat2_final * surf3s_final) / np.sum(surf3s_final ** 2)
    Natb = Nat - ratio2 * best_params['surf2s']


    # --- ファイル保存処理 ---
    output_test_path = output_dir / f"{base_filename}.test.dat"
    output_data1 = np.column_stack([wl, Nat / np.mean(Nat2_final),
                                    best_params['surf2s'] * best_params['ratiof_full'] / np.mean(Nat2_final)])
    np.savetxt(output_test_path, output_data1, fmt='%.8e',
               header="Wavelength(nm) Observed_Norm Combined_Solar_Model_Norm")

    try:
        hap = fits.getdata(hapke_path)
        tothap = np.sum(hap) * Shap * dwl * 1e+12
        cts2MR = tothap / ratio2

        output_exos_path = output_dir / f"{base_filename}.exos.dat"
        exos_data = np.column_stack([wl, Natb * cts2MR])
        np.savetxt(output_exos_path, exos_data, fmt='%.8e', header="Wavelength(nm) Flux(MR)")
        print(f"    -> Saved final spectrum to: {output_exos_path.name}")
    except FileNotFoundError:
        print(f"    -> WARNING: Hapke file not found at {hapke_path}. Skipping final conversion.")
    except Exception as e:
        print(f"    -> ERROR: An error occurred during final conversion: {e}")

# ==============================================================================
# スクリプトの実行部
# ==============================================================================
if __name__ == "__main__":
    # --- 1. 基本設定 ---
    day = "20250825"
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    data_dir = base_dir / "output" / day
    csv_file_path = base_dir / "2025ver" / f"mcparams{day}.csv"
    solar_spec_path = base_dir / "SolarSpectrum.txt"

    TYPES_TO_PROCESS = ['MERCURY']
    type_col = 'Type'

    # ★★★【設定項目】★★★
    # --- 2. フィッティング設定 ---
    FIT_CONFIG = {
        # --- d2輝線周辺のフィット除外 ---
        'exclude_d2_region': True,
        'd2_wavelength_nm': 589.7558,
        'd2_exclusion_half_width_pix': 20,

        # --- 全体のフィッティング波長範囲の手動指定 ---
        'wavelength_range_nm': [589.15,590.0],

        # --- 特定の波長領域をフィットから除外するリスト ---
        'exclusion_wavelength_ranges_nm': None,

        # ★★★【新機能】波長の強制アラインメント ★★★
        # 指定した太陽光と観測スペクトルの波長が一致するようにsft値を自動計算します。
        # この設定を有効にすると、下の `sft_values_to_test` は無視されます。
        # 不要な場合は None に設定します。
        #'force_align_wavelengths': {
        #    'solar_nm': 590.182,     # 太陽光スペクトルでの基準となる波長 (例: Fraunhofer D2線)
        #    'observed_nm': 590.119,  # 観測スペクトルでの基準となる波長 (例: Na輝線)
        #},
         'force_align_wavelengths': None, # 無効にする場合

        # --- その他 ---
        'create_debug_plot': False,
    }

    # ★★★ このリストは、強制アラインメントが無効の場合に使われます ★★★
    sft_values_to_test_default = [-0.0005, 0.0005, 0.0015] #dusk

    # ★★★ アラインメント設定に基づいてsft値を決定 (★★★ ロジック変更箇所 ★★★) ★★★
    align_config = FIT_CONFIG.get('force_align_wavelengths')
    if align_config and 'solar_nm' in align_config and 'observed_nm' in align_config:
        solar_ref_wl = align_config['solar_nm']
        obs_ref_wl = align_config['observed_nm']
        # sft = solar_original - observed_target となるように計算
        calculated_sft = solar_ref_wl - obs_ref_wl
        sft_values_to_test = [calculated_sft]
        print("\n" + "#" * 60)
        print("  波長強制アラインメントが有効です。")
        print(f"  太陽光 {solar_ref_wl} nm が 観測 {obs_ref_wl} nm に一致するように、")
        print(f"  sft = {calculated_sft:.6f} の単一値で処理を実行します。")
        print("#" * 60)
    else:
        sft_values_to_test = sft_values_to_test_default
        print("\n" + "#" * 60)
        print("  波長強制アラインメントは無効です。")
        print(f"  指定されたsft値のリストで処理を実行します: {sft_values_to_test}")
        print("#" * 60)


    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()

    for sft_val in sft_values_to_test:
        print("\n" + "=" * 25 + f" sft = {sft_val:.6f} の処理を開始 " + "=" * 25)

        for process_type in TYPES_TO_PROCESS:
            print("\n" + "-" * 20 + f" 処理タイプ: {process_type} " + "-" * 20)
            target_df = df[df[type_col] == process_type].copy()
            if target_df.empty:
                print(f"-> CSV内に '{process_type}' のデータが見つかりませんでした。")
                continue

            for idx, (row_index, row) in enumerate(target_df.iterrows(), start=1):
                base_name = f"{process_type}{idx}_tr"

                input_file = data_dir / f"{base_name}.totfib.dat"
                if not input_file.exists():
                    input_file = data_dir / f"{base_name}.totfib_orig.dat"

                if not input_file.exists():
                    print(f"  -> スキップ: 入力ファイル {input_file.name} が見つかりません。")
                    continue

                try:
                    hapke_path = data_dir / f"Hapke{day}.fits"
                    constants_this_run = {
                        'Vme': row['mercury_earth_radial_velocity_km_s'],
                        'Vms': row['mercury_sun_radial_velocity_km_s'],
                        'Rmn': row['apparent_diameter_arcsec'],
                        'sft': sft_val,
                        'Rmc': 4.879e+8,
                        'c': 299792.458
                    }
                except KeyError as e:
                    print(f"    -> エラー: CSVに列 '{e}' が見つかりません。")
                    continue

                process_spectrum_original_logic(
                    input_dat_path=input_file,
                    solar_spec_path=solar_spec_path,
                    hapke_path=hapke_path,
                    output_dir=data_dir,
                    constants=constants_this_run,
                    fit_config=FIT_CONFIG
                )

    print("\n--- 全ての処理が完了しました ---")