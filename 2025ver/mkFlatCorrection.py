import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import os
import re

# --- パス設定 ---
date = "20250501"
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
# 前処理済みデータが格納されているディレクトリ
input_data_dir = base_dir / f"output/{date}"
# このスクリプトの出力先ディレクトリ
output_dir = base_dir / f"output/{date}"
# 基準となるCSVファイル
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

# --- 1. マスターフラットの作成 (CSVからLEDファイルを自動特定) ---
print('Creating master flat...')

# CSVからLEDファイルのみを抽出
led_df = df[df[desc_col] == 'LED']
if led_df.empty:
    print("エラー: CSV内に 'LED' のファイルが見つかりません。マスターフラットを作成できません。")
    exit()


# 番号を抽出する関数
def get_file_number(filename_stem, index):
    match = re.search(r'_(\d+)$', filename_stem)
    return match.group(1) if match else str(index + 1)


# 基準となるファイルから配列の形状を取得
try:
    # 最初のLEDファイルを使って形状を決定
    first_led_row = led_df.iloc[0]
    first_led_desc = first_led_row[desc_col]
    first_led_stem = Path(first_led_row[fits_col]).stem
    first_led_num = get_file_number(first_led_stem, 0)  # インデックスは仮

    template_filename = f"{first_led_desc}{first_led_num}_tr.fits"
    with fits.open(input_data_dir / template_filename) as hdul:
        template_shape = hdul[0].data.shape
except FileNotFoundError:
    print(f"エラー: 基準となるフラットファイル {template_filename} が見つかりません。")
    exit()

# マスターフラットを倍精度(float64)で初期化
master_flat = np.zeros(template_shape, dtype=np.float64)

# 抽出したLEDファイルを全て足し合わせる
for index, row in led_df.iterrows():
    description = row[desc_col]
    stem = Path(row[fits_col]).stem
    file_num = get_file_number(stem, index)

    flat_filename = f"{description}{file_num}_tr.fits"
    flat_path = input_data_dir / flat_filename
    print(f"  Adding flat frame: {flat_path.name}")
    try:
        flat_data = fits.getdata(flat_path, ext=0).astype(np.float64)
        master_flat += flat_data
    except FileNotFoundError:
        print(f"  警告: フラットファイル {flat_path.name} が見つかりませんでした。スキップします。")

# --- 2. マスターフラットの規格化（ノーマライズ） ---
row_idx, col_idx = 72, 642
norm_value = master_flat[row_idx, col_idx]
print(f"Normalizing master flat with value at (row={row_idx}, col={col_idx}): {norm_value}")

if norm_value != 0:
    master_flat /= norm_value
else:
    print("警告: 規格化ピクセルの値がゼロです。規格化をスキップします。")

master_flat_path = output_dir / "flat.fit"
fits.writeto(master_flat_path, master_flat, overwrite=True)
print(f"Master flat saved to: {master_flat_path}")

# --- 3. 科学データのフラット補正 (LED以外の全ファイルを処理) ---
print('\nApplying flat field correction to science frames...')
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

        corrected_data = np.divide(science_data, master_flat,
                                   out=np.zeros_like(science_data),
                                   where=master_flat != 0)

        # 補正後のデータを保存
        output_filename = f"{description}{file_num}_f.fits"
        output_path = output_dir / output_filename
        fits.writeto(output_path, corrected_data, overwrite=True)

    except FileNotFoundError:
        print(f"  警告: 科学データ {science_path.name} が見つかりませんでした。スキップします。")

print('\nFlat fielding completed.')