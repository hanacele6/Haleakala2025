import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import warnings


def combine_fits_files(file_list, save_path, combine_type='clippedmean', sigma=2.0, make_plots=True, plot_dir=None):
    """
    複数のFITSファイルを読み込み、1つに合成して保存する関数。
    """
    if not file_list:
        return False

    print(f"  > {len(file_list)} 個のファイルを '{combine_type}' で合成中...")

    all_data = []
    header_info = {'expt': [], 'jdmid': [], 'ctemp': []}
    base_header = None

    for f_path in file_list:
        try:
            with fits.open(f_path) as hdul:
                data = hdul[0].data.astype(np.float64)
                header = hdul[0].header
                if base_header is None:
                    base_header = header  # 最初のヘッダーを基準とする

                all_data.append(data)

                # ヘッダー情報を収集
                header_info['expt'].append(header.get('EXPOSURE', 0))
                obs_date = header.get('DATE-OBS', header.get('DATE'))
                if obs_date:
                    try:
                        t = Time(obs_date, format='isot', scale='utc')
                        header_info['jdmid'].append(t.jd + header.get('EXPOSURE', 0) / 86400 / 2)
                    except ValueError:
                        pass  # 時刻フォーマットエラー時はスキップ
                header_info['ctemp'].append(float(header.get('CCD-TEMP', 0)))

        except Exception as e:
            print(f"  > 警告: ファイル {f_path.name} の読み込みエラー: {e}")

    if not all_data:
        print("  > エラー: 有効なFITSデータが読み込めませんでした。")
        return False

    # NumPy配列に変換
    data_cube = np.array(all_data)

    # 指定された方法で合成 (ここで警告をミュートする)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=".*'partition' will ignore the 'mask'.*")

        if combine_type == 'clippedmean':
            masked_array = sigma_clip(data_cube, sigma=sigma, axis=0)
            combined_data = np.mean(masked_array, axis=0)
        elif combine_type == 'median':
            combined_data = np.median(data_cube, axis=0)
        else:
            combined_data = np.mean(data_cube, axis=0)

    # ヘッダー情報の更新
    if base_header:
        base_header['NCOMBINE'] = (len(all_data), 'Number of combined frames')
        base_header['COMBTYPE'] = (combine_type, 'Frame combination method')
        base_header['HISTORY'] = f'Combined from {len(all_data)} files by pipeline'
        base_header['HISTORY'] = f'Combination performed on {datetime.now().isoformat()}'

        if header_info['expt']: base_header['EXPOSURE'] = np.mean(header_info['expt'])
        if header_info['jdmid']: base_header['DATE-OBS'] = Time(np.mean(header_info['jdmid']), format='jd').isot
        if header_info['ctemp']: base_header['CCD-TEMP'] = (f"{np.mean(header_info['ctemp']):.2f}",
                                                            "Mean CCD Temperature")

    # FITSファイルとして保存（ここでは常に直下に保存し、main.pyの最後で整理させる）
    fits.writeto(save_path, combined_data.astype(np.float32), header=base_header, overwrite=True)
    print(f"  > 合成ファイルを保存しました: {save_path.name}")

    # パイプラインを止めないように画像を保存して閉じる
    if make_plots and plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{save_path.stem}.png"

        plt.figure(figsize=(10, 6))
        vmin, vmax = np.percentile(combined_data, [5, 95])
        plt.imshow(combined_data, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f'Combined Frame: {save_path.name}')
        plt.xlabel('Dispersion Axis (pixels)')
        plt.ylabel('Fiber Axis (pixels)')
        plt.colorbar(label='Intensity')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    return True


def run(run_info, config):
    """
    パイプラインから呼び出される合成処理の実行関数
    """
    output_dir = run_info["output_dir"]
    csv_file_path = run_info["csv_path"]

    # config.yaml から設定を読み込む
    comb_conf = config.get("combination", {})
    target_types = comb_conf.get("target_types", ['LED', 'SKY', 'HLG'])
    combine_type = comb_conf.get("combine_type", 'clippedmean')
    sigma = comb_conf.get("sigma", 2.0)
    make_plots = comb_conf.get("make_plots", True)
    input_suffix = comb_conf.get("input_suffix", '_tr.fits')
    output_suffix = comb_conf.get("output_suffix", '.fits')
    force_rerun = config.get("pipeline", {}).get("force_rerun_combine", False)

    print(f"\n--- マスターフレーム合成処理を開始します ---")

    try:
        df = pd.read_csv(csv_file_path)
        type_col = df.columns[1]
    except Exception as e:
        print(f"エラー: CSVファイルの読み込みに失敗しました: {e}")
        return

    plot_dir = output_dir / "plots"

    for current_type in target_types:
        target_df = df[df[type_col] == current_type]

        if target_df.empty:
            continue

        # 出力ファイル名の決定 (例: master_led.fits)
        output_filename = f"master_{current_type.lower()}{output_suffix}"
        output_filepath = output_dir / output_filename

        # 出力ファイルがすでに整理フォルダ(1_fits)にあるかチェック
        output_filepath_organized = output_dir / "1_fits" / output_filename
        is_processed = output_filepath.exists() or output_filepath_organized.exists()

        # ▼▼▼ スキップ処理 ▼▼▼
        if is_processed and not force_rerun:
            print(f"[{current_type}] 処理済みスキップ: {output_filename}")
            continue

        print(f"\n[{current_type}] 合成処理を開始します...")

        files_to_combine = []
        for file_num, _ in enumerate(target_df.iterrows(), 1):
            input_filename = f"{current_type}{file_num}{input_suffix}"
            file_path = output_dir / input_filename

            # 直下に無ければ整理フォルダ(1_fits)を探す
            if not file_path.exists():
                file_path = output_dir / "1_fits" / input_filename

            if file_path.exists():
                files_to_combine.append(file_path)
            else:
                print(f"  > 警告: 対象ファイルが見つかりません: {input_filename}")

        if files_to_combine:
            combine_fits_files(
                file_list=files_to_combine,
                save_path=output_filepath,
                combine_type=combine_type,
                sigma=sigma,
                make_plots=make_plots,
                plot_dir=plot_dir
            )
        else:
            print(f"  > {current_type} の有効なファイルがありませんでした。")

    print("--- 合成処理が完了しました ---")


if __name__ == '__main__':
    print("このスクリプトは main.py からモジュールとして呼び出してください。")