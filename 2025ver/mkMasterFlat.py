import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.time import Time
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# --------------------------------------------------------------------------
# 設定項目：ご自身の環境に合わせてここを修正してください
# --------------------------------------------------------------------------
# ルートディレクトリ
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
# 日付フォルダ
date = "20250822"
# 観測ログのCSVファイル
csv_file_path = base_dir / "2025ver" / f"mcparams{date}.csv"

# --- 変更点 1: 処理したいデータ種類をリストで指定 ---
TYPES_TO_PROCESS = ['LED', 'SKY']
# --------------------------------------------------------------------------

# --- ディレクトリ設定 ---
# フラット補正済みデータ(_f.fits)が格納されているディレクトリ
input_data_dir = base_dir / f"output/{date}"
# このスクリプトの出力先ディレクトリ
output_dir = base_dir / f"output/{date}"
output_dir.mkdir(parents=True, exist_ok=True)


def combine_fits_files(file_list, save_path, combine_type='clippedmean', sigma=2.0, show_result=True):
    """
    複数のFITSファイルを読み込み、1つに合成して保存する関数。
    """
    if not file_list:
        print("エラー: 処理対象のファイルリストが空です。")
        return

    print(f"{len(file_list)} 個のFITSファイルを '{combine_type}' で合成します...")

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
                    t = Time(obs_date, format='isot', scale='utc')
                    header_info['jdmid'].append(t.jd + header.get('EXPOSURE', 0) / 86400 / 2)
                header_info['ctemp'].append(float(header.get('CCD-TEMP', 0)))

        except FileNotFoundError:
            print(f"  警告: ファイル {f_path.name} が見つかりません。スキップします。")
        except Exception as e:
            print(f"  警告: ファイル {f_path.name} の読み込み中にエラー: {e}。スキップします。")

    if not all_data:
        print("エラー: 有効なFITSデータが読み込めませんでした。処理を終了します。")
        return

    # NumPy配列に変換
    data_cube = np.array(all_data)

    # 指定された方法で合成
    if combine_type == 'clippedmean':
        masked_array = sigma_clip(data_cube, sigma=sigma, axis=0)
        combined_data = np.mean(masked_array, axis=0)
    elif combine_type == 'median':
        combined_data = np.median(data_cube, axis=0)
    else:  # 'mean' or default
        combined_data = np.mean(data_cube, axis=0)

    # ヘッダー情報の更新
    if base_header:
        base_header['NCOMBINE'] = (len(all_data), 'Number of combined frames')
        base_header['COMBTYPE'] = (combine_type, 'Frame combination method')
        base_header['HISTORY'] = f'Combined from {len(all_data)} files by this script'
        base_header['HISTORY'] = f'Combination performed on {datetime.now().isoformat()}'

        # 収集した情報でヘッダーを更新
        if header_info['expt']: base_header['EXPOSURE'] = np.mean(header_info['expt'])
        if header_info['jdmid']: base_header['DATE-OBS'] = Time(np.mean(header_info['jdmid']), format='jd').isot
        if header_info['ctemp']: base_header['CCD-TEMP'] = (f"{np.mean(header_info['ctemp']):.2f}",
                                                            "Mean CCD Temperature")

    # FITSファイルとして保存
    fits.writeto(save_path, combined_data.astype(np.float32), header=base_header, overwrite=True)
    print(f"-> 合成ファイルを保存しました: {save_path.name}")

    # 結果を表示
    if show_result:
        plt.figure(figsize=(10, 6))
        vmin, vmax = np.percentile(combined_data, [5, 95])
        plt.imshow(combined_data, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f'Combined Frame: {save_path.name}')
        plt.xlabel('Dispersion Axis (pixels)')
        plt.ylabel('Fiber Axis (pixels)')
        plt.colorbar(label='Intensity')
        plt.show()


if __name__ == '__main__':
    print(f"--- FITS合成処理を開始します (日付: {date}) ---")

    # --- CSVの読み込み ---
    try:
        df = pd.read_csv(csv_file_path)
        fits_col = df.columns[0]
        type_col = df.columns[1]
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        sys.exit()
    except IndexError:
        print(f"エラー: CSVファイルに2列以上のデータがありません。")
        sys.exit()

    # --- 変更点 2: リスト内の各タイプに対して処理を繰り返す ---
    for current_type in TYPES_TO_PROCESS:
        print(f"\n--- 処理を開始します: [ {current_type} ] ---")

        # --- 変更点 3: 'current_type' を使ってCSVをフィルタリング ---
        target_df = df[df[type_col] == current_type]

        if target_df.empty:
            print(f"警告: CSV内に '{current_type}' のデータが見つかりませんでした。スキップします。")
            continue # ループの次のイテレーションへ

        files_to_combine = []
        print("処理対象のファイルを検索します...")
        for file_num, (index, row) in enumerate(target_df.iterrows(), 1):
            file_type = row[type_col]
            # 例: 1番目のSKYファイルなら "SKY1_tr.fits"
            input_filename = f"{file_type}{file_num}_tr.fits"
            # input_filename = f"{file_type}{file_num}_f.fits" #感度校正後はこっち

            file_path = input_data_dir / input_filename
            files_to_combine.append(file_path)
            print(f"  + {input_filename}  (連番 {file_num} を使用)")

        # --- 変更点 4: 'current_type' を使って出力ファイル名を決定 ---
        output_filepath = output_dir / f"master_{current_type.lower()}.fits"
        #output_filepath = output_dir / f"master_{current_type.lower()}_f.fits" #感度校正後はこっち

        # FITS合成の実行
        combine_fits_files(
            file_list=files_to_combine,
            save_path=output_filepath
        )

    print("\n--- 全ての処理が完了しました ---")