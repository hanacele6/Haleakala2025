import numpy as np
from astropy.io import fits
from pathlib import Path
import pandas as pd


# ==============================================================================
# メインの背景光減算・切り出し関数 (アルゴリズム変更なし)
# ==============================================================================
def subtract_background(input_wc_path, fiber_definitions, output_dir,
                        crop_spectrum=False, center_wl=None, crop_half_width=None):
    """
    波長校正済みの2Dスペクトルを読み込み、背景光減算後、任意でスペクトルを切り出す。
    """
    print(f"  -> Processing: {input_wc_path.name}")

    try:
        with fits.open(input_wc_path) as hdul:
            spec_2d = hdul[0].data.astype(np.float64)
            header = hdul[0].header
            if 'IFIBERS' in hdul:
                active_fibers = hdul['IFIBERS'].data
            else:
                print("    -> WARNING: 'IFIBERS' extension not found. Assuming all fibers are active.")
                active_fibers = np.arange(spec_2d.shape[0])
    except FileNotFoundError:
        print(f"    -> ERROR: File not found: {input_wc_path}. Skipping.")
        return
    except Exception as e:
        print(f"    -> ERROR: Could not read FITS file {input_wc_path}: {e}. Skipping.")
        return

    try:
        nx = header['NAXIS1']
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        crpix1 = header.get('CRPIX1', 1.0)
        wavelengths = crval1 + (np.arange(nx) - (crpix1 - 1)) * cdelt1
    except KeyError as e:
        print(f"    -> ERROR: WCS keyword {e} not found. Cannot build wavelength axis. Skipping.")
        return

    n_fib_x = fiber_definitions['NFIBX']
    target_rows = fiber_definitions['target_rows']
    background_rows = fiber_definitions['background_rows']
    all_target_fibers = [jx + jy * n_fib_x for jy in target_rows for jx in range(n_fib_x)]
    all_bkg_fibers = [jx + jy * n_fib_x for jy in background_rows for jx in range(n_fib_x)]
    valid_target_fibers = np.intersect1d(all_target_fibers, active_fibers)
    valid_bkg_fibers = np.intersect1d(all_bkg_fibers, active_fibers)
    n_target, n_bkg = len(valid_target_fibers), len(valid_bkg_fibers)

    if n_target == 0 or n_bkg == 0:
        print("    -> WARNING: Not enough valid target or background fibers. Skipping.")
        return

    target_sum_spec = spec_2d[valid_target_fibers, :].sum(axis=0)
    bkg_sum_spec = spec_2d[valid_bkg_fibers, :].sum(axis=0)
    bkg_per_fiber_spec = bkg_sum_spec / n_bkg
    final_spectrum = target_sum_spec - (bkg_per_fiber_spec * n_target)

    base_filename = input_wc_path.name.replace(".wc.fits", "")

    if crop_spectrum:
        if center_wl is None or crop_half_width is None:
            print("    -> ERROR: Cropping is enabled, but center wavelength/width is not set. Skipping.")
            return

        center_idx = np.argmin(np.abs(wavelengths - center_wl))
        start_idx = center_idx - crop_half_width
        end_idx = center_idx + crop_half_width + 1

        if start_idx < 0 or end_idx > len(wavelengths):
            print(f"    -> ERROR: Cannot crop around {center_wl} nm. Exceeds boundaries.")
            return

        wavelengths_to_save = wavelengths[start_idx:end_idx]
        spectrum_to_save = final_spectrum[start_idx:end_idx]
        output_dat_path = output_dir / f"{base_filename}.totfib.dat"
    else:
        wavelengths_to_save = wavelengths
        spectrum_to_save = final_spectrum
        output_dat_path = output_dir / f"{base_filename}.totfib_orig.dat"

    data_to_save = np.vstack((wavelengths_to_save, spectrum_to_save)).T
    np.savetxt(output_dat_path, data_to_save, fmt='%.8e', header="Wavelength(nm) Intensity", comments='')
    print(f"    -> Saved final spectrum to: {output_dat_path.name}")


# ==============================================================================
# パイプライン実行用モジュール
# ==============================================================================
def run(run_info, config):
    """
    パイプラインから呼び出される背景光減算・スペクトル切り出しの実行関数
    """
    output_dir = run_info["output_dir"]
    csv_file_path = run_info["csv_path"]

    sub_conf = config.get("subtraction", {})
    target_types = sub_conf.get("target_types", ['MERCURY'])
    fiber_defs = sub_conf.get("fiber_defs", {})
    crop_conf = sub_conf.get("crop", {})

    crop_enabled = crop_conf.get("enabled", True)
    center_wl = crop_conf.get("center_wl", 589.7558)
    crop_half_width = crop_conf.get("half_width", 200)

    force_rerun = config.get("pipeline", {}).get("force_rerun_subtract", False)

    print(f"\n--- ファイバー合成・背景光減算処理を開始します ---")
    if crop_enabled:
        print(f"  > 切り出しモードON (中心: {center_wl} nm, 幅: ±{crop_half_width} pix)")

    try:
        df = pd.read_csv(csv_file_path)
        type_col = df.columns[1]
    except Exception as e:
        print(f"エラー: CSV読み込み失敗: {e}")
        return

    for process_type in target_types:
        target_df = df[df[type_col] == process_type]
        if target_df.empty: continue

        print(f"\n[{process_type}] {len(target_df)} 個のファイルを処理します...")

        for i, _ in enumerate(target_df.iterrows(), start=1):
            base_name = f"{process_type}{i}_tr"

            # ★ 修正1: 入力ファイル (*.wc.fits) を直下と 1_fits/ の両方から探す
            input_file_name = f"{base_name}.wc.fits"
            input_file = output_dir / input_file_name
            if not input_file.exists():
                input_file = output_dir / "1_fits" / input_file_name

            # ★ 修正2: 出力ファイル (*.dat) が直下または 2_spectra/ にあるかチェック
            suffix = ".totfib.dat" if crop_enabled else ".totfib_orig.dat"
            output_name = f"{base_name}{suffix}"
            is_processed = (output_dir / output_name).exists() or (output_dir / "2_spectra" / output_name).exists()

            # ▼▼▼ スキップ判定 ▼▼▼
            if is_processed and not force_rerun:
                print(f"  > 処理済みスキップ: {output_name}")
                continue

            if not input_file.exists():
                print(f"  > 警告: 入力ファイルが見つかりません: {input_file_name}")
                continue

            subtract_background(
                input_wc_path=input_file,
                fiber_definitions=fiber_defs,
                output_dir=output_dir,
                crop_spectrum=crop_enabled,
                center_wl=center_wl,
                crop_half_width=crop_half_width
            )

    print("--- ファイバー合成・背景光減算処理が完了しました ---")


if __name__ == "__main__":
    print("このスクリプトは main.py からモジュールとして呼び出してください。")