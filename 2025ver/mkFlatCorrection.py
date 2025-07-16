import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import os
import re

# --- パスとCSV設定 ---
date = "20250501"  # 元のコードに合わせて日付を修正
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
# スペクトル抽出後のデータが格納されているディレクトリ
input_data_dir = base_dir / f"output/{date}"
# このスクリプトの出力先ディレクトリ
output_dir = base_dir / f"output/{date}"
# 基準となるCSVファイル
csv_file_path = Path("mcparams202505.csv")  # CSVファイル名はご自身のものに合わせてください

# --- 出力ディレクトリの作成 ---
output_dir.mkdir(parents=True, exist_ok=True)

# --- CSVファイルの読み込み ---
try:
    df = pd.read_csv(csv_file_path)
    fits_col, desc_col = df.columns[0], df.columns[1]
except (FileNotFoundError, IndexError, KeyError) as e:
    print(f"エラー: CSVファイルが見つからないか、形式が正しくありません: {e}")
    exit()

# --- 1. マスターフラットの作成 (1Dデータを使用) ---
print('Creating master flat from 1D spectra ("LED" files)...')

# CSVからLEDファイルのみを抽出
led_df = df[df[desc_col] == 'LED']
if led_df.empty:
    print("エラー: CSV内に 'LED' のファイルが見つかりません。")
    exit()


# 番号を抽出する関数
def get_file_number(filename_stem, index):
    match = re.search(r'_(\d+)$', filename_stem)
    return match.group(1) if match else str(index + 1)


# 基準となるファイルから配列の形状を取得
master_flat = None

# 抽出したLEDファイルを全て足し合わせる
for index, row in led_df.iterrows():
    description = row[desc_col]
    stem = Path(row[fits_col]).stem
    file_num = get_file_number(stem, index)

    # １次元スペクトルファイル (_tr.fits) を読み込む
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

fits.writeto(output_dir / "debug_master_flat_BEFORE_NORM.fit", master_flat, overwrite=True)
print("DEBUG: Saved master_flat before normalization.")


if master_flat is None:
    print("エラー: 有効なLEDファイルが1つも見つからなかったため、処理を終了します。")
    exit()

# --- 2. マスターフラットの規格化（元のコードのロジックを維持） ---
# 1Dデータなので、座標は (ファイバー番号, 波長ピクセル) を意味する
row_idx, col_idx = 55, 942
norm_value = master_flat[row_idx, col_idx]
print(f"DEBUG: The normalization value at (fiber={row_idx}, pix={col_idx}) is: {norm_value}")
print(f"Normalizing master flat with value at (fiber={row_idx}, wavelength_pix={col_idx}): {norm_value}")

if norm_value != 0:
    master_flat /= norm_value
else:
    print("警告: 規格化ピクセルの値がゼロです。規格化をスキップします。")

master_flat_path = output_dir / "flat.fit"
fits.writeto(master_flat_path, master_flat, overwrite=True)
print(f"Master flat saved to: {master_flat_path}")

# --- 2. マスターフラットの規格化（ファイバーごとに行う） ---
#print('Normalizing master flat fiber by fiber...')

# 新しいファイバーごとの規格化ループ
#for i in range(master_flat.shape[0]):
    # 1本分のファイバーのデータを取り出す
#    fiber_data = master_flat[i, :]

    # ゼロやマイナスの値を除いた、そのファイバーの中央値を取得
    # これにより、ノイズや異常値に強い、安定した規格化が可能
#    valid_pixels = fiber_data[fiber_data > 0]
#    if valid_pixels.size > 0:
#        norm_value = np.median(valid_pixels)
#    else:
        # 有効なピクセルがない場合はスキップ
#        continue

    # ゼロ除算を避ける
#    if norm_value != 0:
        # そのファイバーのデータだけを、そのファイバー自身の規格化値で割る
#        master_flat[i, :] /= norm_value

#print("Fiber-by-fiber normalization completed.")

# 完成したマスターフラットを保存
#master_flat_path = output_dir / "flat.fit"
#fits.writeto(master_flat_path, master_flat, overwrite=True)
#print(f"Master flat saved to: {master_flat_path}")

# --- 3. 科学データのフラット補正 (LED以外の全ファイルを処理) ---
print('\nApplying flat field correction to science frames...')
science_df = df[df[desc_col] != 'LED']

for index, row in science_df.iterrows():
    description = row[desc_col]
    stem = Path(row[fits_col]).stem
    file_num = get_file_number(stem, index)

    # 科学データの1次元スペクトルファイル (_tr.fits) を読み込む
    science_filename_in = f"{description}{file_num}_tr.fits"
    science_path = input_data_dir / science_filename_in
    print(f"  Processing: {science_path.name}")

    try:
        science_data = fits.getdata(science_path, ext=0).astype(np.float64)
        header = fits.getheader(science_path, ext=0)

        # 1次元データ同士で割り算を実行して補正
        corrected_data = np.divide(science_data, master_flat,
                                   out=np.zeros_like(science_data),
                                   where=master_flat != 0)

        # 補正後のデータを保存
        output_filename = f"{description}{file_num}_f.fits"
        output_path = output_dir / output_filename
        fits.writeto(output_path, corrected_data, header=header, overwrite=True)

    except FileNotFoundError:
        print(f"  警告: 科学データ {science_path.name} が見つかりませんでした。スキップします。")

print('\nFlat fielding completed.')