import numpy as np
from astropy.io import fits
from pathlib import Path
import sys
import pandas as pd

def subtract_background(input_wc_path, fiber_definitions, output_dir,
                          crop_spectrum=False, center_wl=None, crop_half_width=None): # ★★★ 引数を追加
    """
    波長校正済みの2Dスペクトルを読み込み、背景光減算後、任意でスペクトルを切り出す。

    Args:
        input_wc_path (Path): 入力となる波長校正済みFITSファイルのパス (*.wc.fits)。
        fiber_definitions (dict): ファイバーのジオメトリとグループ分けを定義した辞書。
        output_dir (Path): 出力ファイルを保存するディレクトリ。
        crop_spectrum (bool, optional): Trueの場合、スペクトルを切り出す。Defaults to False.
        center_wl (float, optional): 切り出す中心波長(nm)。 Defaults to None.
        crop_half_width (int, optional): 中心から左右に切り出すデータ点数。Defaults to None.
    """
    print(f"  -> Processing: {input_wc_path.name}")

    # --- FITSファイルの読み込み (変更なし) ---
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

    # --- 波長軸の再構築 (変更なし) ---
    try:
        nx = header['NAXIS1']
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        crpix1 = header.get('CRPIX1', 1.0)
        wavelengths = crval1 + (np.arange(nx) - (crpix1 - 1)) * cdelt1
    except KeyError as e:
        print(f"    -> ERROR: WCS keyword {e} not found. Cannot build wavelength axis. Skipping.")
        return

    # --- ファイバーのグループ分けと背景光減算 (変更なし) ---
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
    print(f"    -> Found {n_target} valid target fibers and {n_bkg} valid background fibers.")
    target_sum_spec = spec_2d[valid_target_fibers, :].sum(axis=0)
    bkg_sum_spec = spec_2d[valid_bkg_fibers, :].sum(axis=0)
    bkg_per_fiber_spec = bkg_sum_spec / n_bkg
    final_spectrum = target_sum_spec - (bkg_per_fiber_spec * n_target)

    # --- ★★★ ここから追加：スペクトルの切り出し処理 ★★★ ---
    base_filename = input_wc_path.name.replace(".wc.fits", "")

    if crop_spectrum:
        if center_wl is None or crop_half_width is None:
            print("    -> ERROR: Cropping is enabled, but center wavelength/width is not set. Skipping.")
            return

        # 1. 中心波長に最も近いデータのインデックスを探す
        center_idx = np.argmin(np.abs(wavelengths - center_wl))

        # 2. 切り出す範囲の開始・終了インデックスを計算
        start_idx = center_idx - crop_half_width
        end_idx = center_idx + crop_half_width + 1  # Pythonのスライスは最後の要素を含まないため+1

        # 3. 配列の範囲外にならないかチェック
        if start_idx < 0 or end_idx > len(wavelengths):
            print(f"    -> ERROR: Cannot crop around {center_wl} nm.")
            print(f"       The window [{start_idx}:{end_idx}] exceeds data boundaries [0:{len(wavelengths)}]. Skipping.")
            return

        # 4. 波長とスペクトルを切り出す
        wavelengths_to_save = wavelengths[start_idx:end_idx]
        spectrum_to_save = final_spectrum[start_idx:end_idx]
        output_dat_path = output_dir / f"{base_filename}.totfib.dat" # ファイル名変更
        print(f"    -> Cropped spectrum to {len(wavelengths_to_save)} points around {center_wl:.2f} nm.")

    else:
        # 切り出しを行わない場合は、元のデータをそのまま使う
        wavelengths_to_save = wavelengths
        spectrum_to_save = final_spectrum
        output_dat_path = output_dir / f"{base_filename}.totfib_orig.dat"

    # --- 結果をテキストファイルに保存 ---
    data_to_save = np.vstack((wavelengths_to_save, spectrum_to_save)).T
    np.savetxt(output_dat_path, data_to_save, fmt='%.8e', header="Wavelength(nm) Intensity", comments='')
    print(f"    -> Saved final spectrum to: {output_dat_path.name}")


# ==============================================================================
# スクリプトの実行部
# ==============================================================================
if __name__ == "__main__":
    # --- 基本設定（ユーザーが環境に合わせて変更する部分） ---

    # 1. & 2. & 3.
    day = "20251021"
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
    data_dir = base_dir / "output" / day
    csv_file_path = base_dir / "2025ver" / f"mcparams{day}.csv"
    TYPES_TO_PROCESS = ['MERCURY']
    type_col = 'Type'
    fiber_defs = {
        'NFIBX': 10,
        'NFIBY': 12,
        'target_rows': [  3, 4, 5],#水星あるとこ
        'background_rows': [0, 1,2 ,6,7,8, 9, 10, 11]#水星ないとこ
        #'target_rows': [2,3, 4, 5, 6, 7,8,9,10],  # 水星あるとこ
        #'background_rows': [0, 1, 11]  # 水星ないとこ
    }

    # 4. ★★★ スペクトル切り出し設定 ★★★
    CROP_SPECTRUM = True  # Trueにするとスペクトルを切り出す
    # 切り出す中心の波長(nm)を指定 (例: ナトリウムD線)
    CENTER_WAVELENGTH = 589.7558  #真空
    #CENTER_WAVELENGTH = 589.9  # 真空
    #CENTER_WAVELENGTH = 589.594   #空気中
    # 中心から左右に切り出すデータ点数 (±200点 -> 合計401点)
    CROP_HALF_WIDTH = 200

    # --- 処理の開始 (変更なし) ---
    print("--- ファイバー合成・背景光減算処理を開始します ---")
    if CROP_SPECTRUM:
        print(f"*** スペクトル切り出しモードON (中心: {CENTER_WAVELENGTH} nm, 幅: ±{CROP_HALF_WIDTH} pix) ***")
    print(f"データディレクトリ: {data_dir}")

    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()

    for process_type in TYPES_TO_PROCESS:
        print("\n" + "=" * 25 + f" 処理タイプ: {process_type} " + "=" * 25)
        target_df = df[df[type_col] == process_type].copy()
        if target_df.empty:
            print(f"-> CSV内にタイプ '{process_type}' のデータは見つかりませんでした。")
            continue
        print(f"-> {len(target_df)}個の '{process_type}' ファイルを処理します...")

        for i, (index, row) in enumerate(target_df.iterrows(), start=1):
            base_name = f"{process_type}{i}_tr"
            input_file = data_dir / f"{base_name}.wc.fits"
            if not input_file.exists():
                print(f"  -> スキップ: 入力ファイル {input_file.name} が見つかりません。")
                continue

            # ★★★ メインの処理関数に新しい引数を渡す ★★★
            subtract_background(
                input_wc_path=input_file,
                fiber_definitions=fiber_defs,
                output_dir=data_dir,
                crop_spectrum=CROP_SPECTRUM,
                center_wl=CENTER_WAVELENGTH,
                crop_half_width=CROP_HALF_WIDTH
            )

    print("\n--- 全ての処理が完了しました ---")