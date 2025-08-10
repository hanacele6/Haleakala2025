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
    """
    sft_val = constants['sft']
    sft_suffix = f"_sft{int(sft_val * 10000):03d}"  # 例: sft=0.001 -> _sft01
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

    # 太陽光の波長を真空→大気に変換
    #wavair_factor = 1.000276
    wavair_factor = 1.000
    sol_data[:, 0] = sol_data[:, 0] / wavair_factor

    # モデル計算に必要な2種類の太陽光スペクトルの波長軸を計算
    # 1. 直接光 (sftシフト適用)
    direct_solar_wl = sol_data[:, 0] - sft #- 0.29
    # 2. 水星反射光 (sftシフト + ドップラーシフト適用)
    reflected_solar_wl = direct_solar_wl * (1 + Vms / c) * (1 + Vme / c)

    # --- 3. 処理範囲の決定と観測データのクロップ (コード3の方式) ---
    # 2つの波長軸が両方とも存在する「共通の有効範囲」を計算する
    valid_wl_min = max(np.min(direct_solar_wl), np.min(reflected_solar_wl))
    valid_wl_max = min(np.max(direct_solar_wl), np.max(reflected_solar_wl))
    print(f"    -> Valid overlapping solar range: {valid_wl_min:.4f} - {valid_wl_max:.4f} nm")

    # 計算した有効範囲を使って観測データを切り取る
    original_count = len(wl)
    crop_mask = (wl >= valid_wl_min) & (wl <= valid_wl_max)
    wl, Nat = wl[crop_mask], Nat[crop_mask]

    # 切り取り後のデータ点数を確認
    if len(wl) < 20:
        print("    -> ERROR: No/few overlapping wavelength points found after cropping.")
        return

    ixm = len(wl)
    dwl = np.median(np.diff(wl))
    print(f"    -> Cropped observation data to solar range. Points: {original_count} -> {ixm}")

    # --- 4. 補間用の太陽光モデルの最終準備 ---
    # 後の5番セクションで使う形式にデータを格納し直す
    sol = np.zeros((sol_data.shape[0], 3))
    sol[:, 0] = direct_solar_wl
    sol[:, 1] = sol_data[:, 2]  # フラックス成分1
    sol[:, 2] = sol_data[:, 1]  # フラックス成分2
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

    # --- 6. 最適化ループの準備 (d2除外のON/OFF) ---
    """
    if fit_config['exclude_d2_region']:
        center_pix = ixm // 2
        d2 = fit_config['d2_pixel_index']
        if d2 >= ixm:
            print(f"    -> WARNING: d2 ({d2}) is out of bounds for data width {ixm}. Using center pixel instead.")
            d2 = center_pix
        print(f"    -> Fitting will exclude region around d2={d2}.")

        slice_end1, slice_start1 = center_pix - 7, center_pix + 6
        slice_end2, slice_start2 = d2 - 7, d2 + 6

        #slice_end1, slice_start1 = center_pix - 4, center_pix + 3
        #slice_end2, slice_start2 = d2 - 4, d2 + 3

        if not (0 <= slice_end2 < slice_start2 <= ixm and 0 <= slice_end1 < slice_start1 <= ixm):
            print(f"    -> ERROR: Slicing indices are out of bounds for data width {ixm}. Skipping.")
            return

        Nat2 = np.concatenate((Nat[0:slice_end2], Nat[slice_start2:ixm]))
        pix_range_fit = np.concatenate((np.arange(slice_end2), np.arange(slice_start2, ixm)))
        surf_slicer = lambda s: np.concatenate((s[0:slice_end2], s[slice_start2:ixm]))
    """

    if fit_config['exclude_d2_region']:
        # 1. パラメータを取得
        d2_wl = fit_config['d2_wavelength_nm']
        half_width = fit_config['d2_exclusion_half_width_pix']

        # 2. 波長(wl)配列から、指定したd2_wlに最も近いピクセルのインデックスを動的に見つける
        d2_idx = np.argmin(np.abs(wl - d2_wl))
        print(f"    -> D2 line ({d2_wl} nm) found near pixel index: {d2_idx}")

        # 3. 除外するピクセル範囲を計算する
        exclude_start = max(0, d2_idx - half_width)
        exclude_end = min(ixm, d2_idx + half_width + 1)  # +1 はPythonのスライス仕様のため
        print(f"    -> Excluding pixel range: {exclude_start} to {exclude_end - 1}")

        # 4. 単一のブールマスクを作成し、観測データ、モデル、x軸のすべてに一貫して適用する
        fit_mask = np.ones(ixm, dtype=bool)
        fit_mask[exclude_start:exclude_end] = False

        Nat2 = Nat[fit_mask]
        pix_range_fit = np.arange(ixm)[fit_mask]
        surf_slicer = lambda s: s[fit_mask]
    else:
        print("    -> Fitting will use full spectrum (d2 exclusion is OFF).")
        Nat2 = Nat
        pix_range_fit = np.arange(ixm)
        surf_slicer = lambda s: s

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
        #conv_surf0 = np.fft.fftshift(np.real(np.fft.ifft(fft_surf0 * fft_psf2)))
        #conv_surf1 = np.fft.fftshift(np.real(np.fft.ifft(fft_surf1 * fft_psf2)))
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

            #denominator = surf3.copy()
            #zero_mask = np.abs(denominator) < 1e-30
            #denominator[zero_mask] = 1e-30  # ゼロに近い値を微小な値で置換
            #ratioa = Nat2 / denominator

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
    if fit_config['exclude_d2_region']:
        Nat2_final = Nat2
        surf3s_final = surf_slicer(best_params['surf2s'])
    else:
        Nat2_final = Nat
        surf3s_final = best_params['surf2s']

    if np.sum(surf3s_final ** 2) == 0:
        print("    -> ERROR: Cannot calculate ratio2, model sum is zero. Skipping final save.")
        return

    # --- 8. 最終計算と保存 ---

    # ratio2の計算は不要なのでコメントアウトまたは削除
    # if np.sum(surf3s_final ** 2) == 0:
    #     print("    -> ERROR: Cannot calculate ratio2, model sum is zero. Skipping final save.")
    #     return
    # ratio2 = np.sum(Nat2_final * surf3s_final) / np.sum(surf3s_final ** 2)

    # 最適化で見つけた完全なモデルを構築する
    # このモデルは、連続光の形状補正（多項式フィット）を含んでいる
    full_model = best_params['surf2s'] * best_params['ratiof_full']

    # 観測データ全体から、この完全なモデルを引き算する
    Natb = Nat - full_model

    # デバッグ用に、テストファイルに書き出すモデルもfull_modelに変更する
    output_test_path = output_dir / f"{base_filename}.test.dat"
    output_data1 = np.column_stack([wl, Nat / np.mean(Nat),
                                    full_model / np.mean(Nat)])  # ここをfull_modelに変更
    np.savetxt(output_test_path, output_data1, fmt='%.8e',
               header="Wavelength(nm) Observed_Norm Combined_Solar_Model_Norm")

    try:
        hap = fits.getdata(hapke_path)
        tothap = np.sum(hap) * Shap * dwl * 1e+12

        # cts2MRの計算には、モデルの連続光成分を代表する値を使うのが望ましい
        # ここでは、最適化に使われたデータ範囲でのモデルの平均値を使う
        if fit_config['exclude_d2_region']:
            model_for_fit = surf_slicer(full_model)
        else:
            model_for_fit = full_model

        # ゼロ割を避ける
        if np.sum(model_for_fit ** 2) == 0:
            print("    -> ERROR: Model sum is zero. Skipping final conversion.")
            return

        # 再度、観測データとの比を計算して最終的なスケーリング係数を求める
        # これにより、Hapke計算に使う係数がより安定する
        final_scaling_factor = np.sum(Nat2_final * model_for_fit) / np.sum(model_for_fit ** 2)
        cts2MR = tothap / final_scaling_factor

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
    day = "20250712"
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    data_dir = base_dir / "output" / day
    csv_file_path = base_dir / "2025ver" / f"mcparams{day}.csv"
    solar_spec_path = base_dir / "SolarSpectrum.txt"

    TYPES_TO_PROCESS = ['MERCURY']
    type_col = 'Type'

    # ★★★【設定項目】★★★
    # --- 2. フィッティング設定 ---
    FIT_CONFIG = {
        # d2ピクセル周辺のフィット除外を有効にするか (True: 除外する, False: 除外しない)
        'exclude_d2_region': True,

        # 固定ピクセル番号の代わりに、波長と除外する半値幅（ピクセル数）を指定
        #'d2_wavelength_nm': 589.594,  # Na D2線の中心波長 (真空→大気補正後の値に近いもの)
        'd2_wavelength_nm':589.7558,
        'd2_exclusion_half_width_pix': 7,  # 中心から左右に除外するピクセル数

        'create_debug_plot': False
    }

    # 試行するsft値のリスト
    sft_values_to_test = [0.001, 0.002, 0.003]#dawn
    #sft_values_to_test = [-0.0005, 0.0005, 0.0015]#dusk
    #sft_values_to_test = [-0.001, 0.000, 0.001]#test
    #sft_values_to_test = [-0.0015, -0.0005, 0.0005]#test2

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()

        # sft値のリストでループを追加
    for sft_val in sft_values_to_test:
        print("\n" + "#" * 30 + f" sft = {sft_val} の処理を開始 " + "#" * 30)

        for process_type in TYPES_TO_PROCESS:
            print("\n" + "=" * 25 + f" 処理タイプ: {process_type} " + "=" * 25)
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