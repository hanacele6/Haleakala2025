import numpy as np
from astropy.io import fits
from pathlib import Path
import sys
import pandas as pd


def process_spectrum_original_logic(input_dat_path, solar_spec_path, hapke_path, output_dir,
                                    constants):
    """
    元のIDLから移植したPythonコードのロジックを完全に維持し、
    ファイル入出力とパラメータ設定のみを新しい形式に対応させた関数。
    """
    base_filename = input_dat_path.name.split('.')[0]
    print(f"\n  -> Processing: {base_filename} (Original Logic)")

    # --- 1. 入力ファイルの読み込み (新しい形式) ---
    try:
        obs_data = np.loadtxt(input_dat_path, skiprows=1)
        wl = obs_data[:, 0]
        Nat = obs_data[:, 1]
        ixm = len(wl)
        dwl = np.median(np.diff(wl))

        sol_data = np.loadtxt(solar_spec_path)
    except FileNotFoundError as e:
        print(f"    -> ERROR: Input file not found: {e}. Skipping.")
        return
    except Exception as e:
        print(f"    -> ERROR: Failed to read input files: {e}. Skipping.")
        return

    # --- 2. 物理定数とパラメータ (CSVから読み込んだ値を使用) ---
    Vme = constants['Vme']
    Vms = constants['Vms']
    Rmn = constants['Rmn']
    sft = constants['sft']
    Rmc = constants['Rmc']
    c = constants['c']
    Shap = (Rmc / Rmn) ** 2 / 1e+4

    wavair_factor = 1.000276
    # 真空波長を空気波長に変換
    sol_data[:, 0] = sol_data[:, 0] / wavair_factor

    # --- 3. 太陽光モデルの準備 (元のロジックを維持) ---
    sol = np.zeros((sol_data.shape[0], 3))
    sol[:, 0] = sol_data[:, 0]  - sft
    sol[:, 1] = sol_data[:, 2]
    sol[:, 2] = sol_data[:, 1]
    wlsurf = (sol[:, 0] ) * (1 + Vms / c) * (1 + Vme / c)

    # --- 4. 線形補間 (元のforループを維持) ---
    iwm2 = sol.shape[0]
    iws1, iws2 = 0, 0
    surf = np.zeros((ixm, 2), dtype=np.float64)
    for ix in range(ixm):
        # 1列目
        for iw in range(iws1, iwm2 - 1):
            if (wl[ix] - sol[iw, 0]) * (wl[ix] - sol[iw + 1, 0]) <= 0:
                x1, x2 = sol[iw, 0], sol[iw + 1, 0]
                y1, y2 = sol[iw, 1], sol[iw + 1, 1]
                if (x2 - x1) != 0:
                    surf[ix, 0] = (y1 * (x2 - wl[ix]) + y2 * (wl[ix] - x1)) / (x2 - x1)
                else:
                    surf[ix, 0] = y1
                iws1 = iw
                break
        # 2列目
        for iw in range(iws2, iwm2 - 1):
            if (wl[ix] - wlsurf[iw]) * (wl[ix] - wlsurf[iw + 1]) <= 0:
                x1, x2 = wlsurf[iw], wlsurf[iw + 1]
                y1, y2 = sol[iw, 2], sol[iw + 1, 2]
                if (x2 - x1) != 0:
                    surf[ix, 1] = (y1 * (x2 - wl[ix]) + y2 * (wl[ix] - x1)) / (x2 - x1)
                else:
                    surf[ix, 1] = y1
                iws2 = iw
                break

    # --- 5. 最適化ループ (元のロジックを維持) ---
    # スライスルール
    center_pix = ixm // 2
    d2 = 209  # オリジナルコードのハードコード値を維持
    slice_end1 = center_pix - 7
    slice_start1 = center_pix + 6
    slice_end2 = d2 - 7
    slice_start2 = d2 + 6

    Nat2 = np.concatenate((Nat[0:slice_end2], Nat[slice_start2:ixm]))
    zansamin = 1.0e+32
    best_params = {}

    for iFWHM in range(30, 101):
        FWHM = iFWHM * 0.1
        sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # PSFと畳み込み
        psf = np.exp(-((np.arange(ixm, dtype=np.float64) - float(ixm) / 2.0) / sigma) ** 2 / 2.0)
        psf2 = psf / np.sum(psf)
        fft_surf0 = np.fft.fft(surf[:, 0])
        fft_surf1 = np.fft.fft(surf[:, 1])
        fft_psf2 = np.fft.fft(psf2)
        shift_amount = -int(ixm / 2)
        conv_surf0 = np.roll(np.real(np.fft.ifft(fft_surf0 * fft_psf2)), shift=shift_amount)
        conv_surf1 = np.roll(np.real(np.fft.ifft(fft_surf1 * fft_psf2)), shift=shift_amount)
        surf1 = np.column_stack([conv_surf0, conv_surf1])

        for iairm in range(51):
            airm = 0.1 * iairm
            surf2 = surf1[:, 0] ** airm * surf1[:, 1]
            surf3 = np.concatenate((surf2[0:slice_end1], surf2[slice_start1:ixm]))
            pix_range = np.concatenate((np.arange(slice_end1), np.arange(slice_start1, ixm)))

            with np.errstate(divide='ignore'):
                ratioa = Nat2 / surf3

            # polyfitが失敗しないようにチェック
            if not np.all(np.isfinite(ratioa)):
                continue

            aa0 = np.polyfit(pix_range, ratioa, 2)[::-1]
            ratiof = aa0[0] + aa0[1] * pix_range + aa0[2] * pix_range ** 2
            zansa = np.sum((Nat2 - surf3 * ratiof) ** 2)

            if zansa <= zansamin:
                zansamin = zansa
                best_params = {
                    'ratiof_full': aa0[0] + aa0[1] * np.arange(ixm) + aa0[2] * np.arange(ixm) ** 2,
                    'airms': airm,
                    'FWHMs': FWHM,
                    'surf2s': surf2
                }

    if not best_params:
        print("    -> ERROR: Could not find any valid fit parameters. Skipping.")
        return

    print(f"    -> Best fit: airm={best_params['airms']:.2f}, FWHM={best_params['FWHMs']:.2f}, zansa={zansamin:.4e}")

    # --- 6. 最終計算と保存 (元のロジックを維持) ---
    Nat2_final = np.concatenate((Nat[0:slice_end2], Nat[slice_start2:ixm]))
    surf3s_final = np.concatenate((best_params['surf2s'][0:slice_end1], best_params['surf2s'][slice_start1:ixm]))

    # ratio2 の計算
    ratio2 = np.sum(Nat2_final * surf3s_final) / np.sum(surf3s_final * surf3s_final)
    Natb = Nat - ratio2 * best_params['surf2s']

    # 保存処理 (ファイル名は新しい形式)
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
    day = "20250501"
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    data_dir = base_dir / "output" / day
    csv_file_path = base_dir / "2025ver" / f"mcparams{day[:6]}.csv"
    solar_spec_path = base_dir / "SolarSpectrum.txt"

    TYPES_TO_PROCESS = ['MERCURY']
    type_col = 'Type'

    # --- 処理の開始 ---
    print("--- 太陽光スペクトル除去処理を開始します (オリジナルロジック版) ---")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()

    for process_type in TYPES_TO_PROCESS:
        print("\n" + "=" * 25 + f" 処理タイプ: {process_type} " + "=" * 25)
        target_df = df[df[type_col] == process_type].copy()
        if target_df.empty:
            print(f"-> CSV内に '{process_type}' のデータが見つかりませんでした。")
            continue


        # enumerateを使い、カウンター(idx)を1から始める
        for idx, (row_index, row) in enumerate(target_df.iterrows(), start=1):

            # カウンターを使ってファイル名を作成 (MERCURY1_tr, MERCURY2_tr, ...)
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
                    'Rmn': row['mercury_sun_distance_au'],
                    'sft': 0.002,
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
            )

    print("\n--- 全ての処理が完了しました ---")