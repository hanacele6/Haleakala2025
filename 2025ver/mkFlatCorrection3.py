import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import os
import re

# --- パスとCSV設定 ---
date = "20250501"
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
input_data_dir = base_dir / f"output/{date}"
output_dir = base_dir / f"output/{date}"
csv_file_path = Path("mcparams202505.csv")

# --- 出力ディレクトリの作成 ---
output_dir.mkdir(parents=True, exist_ok=True)

# --- CSVファイルの読み込み ---
try:
    df = pd.read_csv(csv_file_path)
    fits_col, desc_col = df.columns[0], df.columns[1]
except (FileNotFoundError, IndexError, KeyError) as e:
    print(f"エラー: CSVファイルが見つからないか、形式が正しくありません: {e}")
    exit()

# --- 1. マスターフラットの作成 (変更なし) ---
print('Creating master flat from 1D spectra ("LED" files)...')
led_df = df[df[desc_col] == 'LED']
if led_df.empty:
    print("エラー: CSV内に 'LED' のファイルが見つかりません。")
    exit()

def get_file_number(filename_stem, index):
    match = re.search(r'_(\d+)$', filename_stem)
    return match.group(1) if match else str(index + 1)

master_flat = None
for index, row in led_df.iterrows():
    description = row[desc_col]
    stem = Path(row[fits_col]).stem
    file_num = get_file_number(stem, index)
    flat_filename = f"{description}{file_num}_tr.fits"
    flat_path = input_data_dir / flat_filename
    print(f"  Adding 1D flat frame: {flat_path.name}")
    try:
        flat_data = fits.getdata(flat_path, ext=0).astype(np.float64)
        if master_flat is None:
            master_flat = flat_data
        else:
            master_flat += flat_data
    except FileNotFoundError:
        print(f"  警告: フラットファイル {flat_path.name} が見つかりませんでした。スキップします。")

if master_flat is None:
    print("エラー: 有効なLEDファイルが1つも見つからなかったため、処理を終了します。")
    exit()

# --- 2. マスターフラットの保存（規格化前） ---
# この時点のマスターフラットをそのまま使います。規格化は補正時に行います。
master_flat_path = output_dir / "master_flat_raw.fit"
fits.writeto(master_flat_path, master_flat, overwrite=True)
print(f"Raw master flat saved to: {master_flat_path}")


# --- 3. 科学データのフラット補正 (### ここからが新しい方法 ###) ---
print('\nApplying new flat field correction to science frames...')
science_df = df[df[desc_col] != 'LED']

for index, row in science_df.iterrows():
    description = row[desc_col]
    stem = Path(row[fits_col]).stem
    file_num = get_file_number(stem, index)

    science_filename_in = f"{description}{file_num}_tr.fits"
    science_path = input_data_dir / science_filename_in
    print(f"  Processing: {science_path.name}")

    try:
        science_data = fits.getdata(science_path, ext=0).astype(np.float64)
        header = fits.getheader(science_path, ext=0)

        # 補正後のデータを入れるための空の配列を準備
        corrected_data = np.zeros_like(science_data)

        # ### 変更点: ファイバーごとに補正ループを実行 ###
        # science_data.shape[0] はファイバーの数
        for i in range(science_data.shape[0]):
            fiber_science = science_data[i, :]
            fiber_flat = master_flat[i, :]

            # ### 変更点: フラットの中央値を計算 ###
            # ヒントのコードのように、スペクトルの安定した範囲で中央値を計算する
            # 例えば、ピクセルインデックス 500から1500 の範囲など。データに合わせて調整してください。
            # この範囲に有効なデータがない場合を考慮
            stable_range_flat = fiber_flat[500:1500]
            valid_pixels = stable_range_flat[stable_range_flat > 0]

            if valid_pixels.size > 0:
                median_flat_value = np.median(valid_pixels)
            else:
                # もし安定領域に有効なピクセルがなければ、ファイバー全体で試す
                valid_pixels_full = fiber_flat[fiber_flat > 0]
                if valid_pixels_full.size > 0:
                    median_flat_value = np.median(valid_pixels_full)
                else:
                    # それでも有効なピクセルがなければ、このファイバーの補正はスキップ
                    print(f"  警告: ファイバー {i} のフラットデータがほぼゼロのため、補正をスキップします。")
                    continue

            # ### 変更点: ゼロ除算を避けながら補正を実行 ###
            # median_flat_value が 0 の場合も考慮
            if median_flat_value > 0:
                 # `where`句でフラット値がゼロに近い場所での割り算を防ぐ
                corrected_fiber = np.divide(
                    fiber_science * median_flat_value, # 科学データに中央値を掛ける
                    fiber_flat,                      # その後、フラットで割る
                    out=np.zeros_like(fiber_science),# ゼロ除算の場合は0を入れる
                    where=(fiber_flat > 1e-6)        # 非常に小さい値での割り算も防ぐ
                )
                corrected_data[i, :] = corrected_fiber

        # 補正後のデータを保存
        output_filename = f"{description}{file_num}_f.fits"
        output_path = output_dir / output_filename
        fits.writeto(output_path, corrected_data, header=header, overwrite=True)

    except FileNotFoundError:
        print(f"  警告: 科学データ {science_path.name} が見つかりませんでした。スキップします。")

print('\nFlat fielding completed.')