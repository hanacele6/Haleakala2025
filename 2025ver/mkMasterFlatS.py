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
date = "20250616"
# 観測ログのCSVファイル
csv_file_path = Path("mcparams20250616.csv")
# 処理したいデータの種類 (CSVの2列目の値)
TYPE_TO_PROCESS = 'SKY'  # 'LED' または 'SKY' を指定
# --------------------------------------------------------------------------

# --- ディレクトリ設定 ---
# 処理対象のデータ(_tr.fits or _f.fits)が格納されているディレクトリ
input_data_dir = base_dir / f"output/{date}"
# このスクリプトの出力先ディレクトリ
output_dir = base_dir / f"output/{date}"
output_dir.mkdir(parents=True, exist_ok=True)


def combine_fits_files(file_list, save_path, combine_type='sum', sigma=2.0, show_result=True):
    """
    複数のFITSファイルを読み込み、1つに合成して保存する関数。
    combine_type に 'sum' を指定すると単純加算を行う。
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
    if combine_type == 'sum':
        # ★★★ ご要望に応じて、ここで単純加算（足し合わせ）を実行 ★★★
        combined_data = np.sum(data_cube, axis=0)
    elif combine_type == 'clippedmean':
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
        base_header['HISTORY'] = f'Combined from {len(all_data)} files by combine_script.py'
        base_header['HISTORY'] = f'Combination performed on {datetime.now().isoformat()}'

        # 収集した情報でヘッダーを更新
        # 加算の場合は露光時間も合計する
        if combine_type == 'sum':
            if header_info['expt']: base_header['EXPOSURE'] = np.sum(header_info['expt'])
        else:  # 平均や中央値の場合は平均露光時間
            if header_info['expt']: base_header['EXPOSURE'] = np.mean(header_info['expt'])

        if header_info['jdmid']: base_header['DATE-OBS'] = Time(np.mean(header_info['jdmid']), format='jd').isot
        if header_info['ctemp']: base_header['CCD-TEMP'] = (f"{np.mean(header_info['ctemp']):.2f}",
                                                            "Mean CCD Temperature")

    # FITSファイルとして保存（データ型は値が大きくなる可能性を考慮しfloat64で保持し、保存時に32にする）
    fits.writeto(save_path, combined_data.astype(np.float32), header=base_header, overwrite=True)
    print(f"-> 合成ファイルを保存しました: {save_path.name}")

    # 結果を表示
    if show_result:
        plt.figure(figsize=(10, 6))
        # 表示レンジを調整して見やすくする
        vmin, vmax = np.percentile(combined_data, [5, 95])
        plt.imshow(combined_data, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        plt.title(f'Combined Frame (Method: {combine_type}): {save_path.name}')
        plt.xlabel('Dispersion Axis (pixels)')
        plt.ylabel('Fiber Axis (pixels)')
        plt.colorbar(label='Total Intensity (Sum)')
        plt.show()


def get_file_number(filename_stem):
    """ファイル名の末尾から連番を抽出する"""
    match = re.search(r'(\d+)$', filename_stem)
    return match.group(1) if match else None


if __name__ == '__main__':
    print("--- FITS加算合成処理を開始します ---")
    print(f"設定:\n  日付: {date}\n  処理タイプ: {TYPE_TO_PROCESS}")

    # --- CSVの読み込みとファイルリストの作成 ---
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

    # 指定されたTYPEのデータのみをフィルタリングする
    target_df = df[df[type_col] == TYPE_TO_PROCESS]

    if target_df.empty:
        print(f"エラー: CSVの2列目({type_col})に '{TYPE_TO_PROCESS}' のデータが見つかりません。")
        sys.exit()

    files_to_combine = []
    print("\n処理対象のファイルを検索します...")
    for index, row in target_df.iterrows():
        stem = Path(row[fits_col]).stem
        file_num = get_file_number(stem)

        input_filename = f"{row[type_col]}{file_num}_tr.fits"
        # input_filename = f"{row[type_col]}{file_num}_f.fits" #感度校正後はこっち
        file_path = input_data_dir / input_filename

        # ファイルが存在するか確認
        if file_path.exists():
            files_to_combine.append(file_path)
            print(f"  + {input_filename}")
        else:
            print(f"  - 警告: ファイル {input_filename} が見つかりませんでした。スキップします。")

    # --- FITS合成の実行 ---
    output_filepath = output_dir / f"master_{TYPE_TO_PROCESS.lower()}_sum.fits"
    # output_filepath = output_dir / f"master_{TYPE_TO_PROCESS.lower()}_sum_f.fits" #感度校正後はこっち

    combine_fits_files(
        file_list=files_to_combine,
        save_path=output_filepath,
        combine_type='sum'  # ★★★ 合成方法として 'sum' (単純加算) を指定 ★★★
    )

    print("\n--- 全ての処理が完了しました ---")