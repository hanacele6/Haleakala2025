import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import re
import sys

# --------------------------------------------------------------------------
# 設定項目：ご自身の環境に合わせてここを修正してください
# --------------------------------------------------------------------------
# ルートディレクトリ
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
# 日付フォルダ
date = "20250614"
# 観測ログのCSVファイル
csv_file_path = Path( "mcparams20250614.csv")
# --------------------------------------------------------------------------

# --- ディレクトリ設定 ---
# スペクトル抽出後のデータ(_tr.fits)が格納されているディレクトリ
input_data_dir = base_dir / f"output/{date}"
# このスクリプトの出力先ディレクトリ
output_dir = base_dir / f"output/{date}"
output_dir.mkdir(parents=True, exist_ok=True)

# --- CSVファイルの読み込み ---
try:
    df = pd.read_csv(csv_file_path)
    # CSVの列名を取得 (ファイル名が1列目、'LED'などの説明が2列目と仮定)
    fits_col, desc_col = df.columns[0], df.columns[1]
except FileNotFoundError:
    print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
    sys.exit()
except (IndexError, KeyError):
    print("エラー: CSVファイルの形式が正しくありません。1列目がファイル名、2列目が説明であることを確認してください。")
    sys.exit()

# --- 1. マスターフラットの作成 ---
# LED(フラット)フレームの生データを単純に合計する
print("Step 1: Creating Raw Master Flat from 'LED' files...")

led_df = df[df[desc_col] == 'LED']
if led_df.empty:
    print("エラー: CSV内に 'LED' のファイルが見つかりません。")
    sys.exit()

raw_master_flat = None
led_file_count = 0


# ファイル名から連番を抽出する関数
def get_file_number(filename_stem):
    match = re.search(r'(\d+)$', filename_stem)
    return match.group(1) if match else None


for index, row in led_df.iterrows():
    stem = Path(row[fits_col]).stem
    file_num = get_file_number(stem)
    if file_num is None:
        print(f"  警告: ファイル名から番号を抽出できませんでした: {stem}")
        continue

    # １次元スペクトルファイル (_tr.fits) を読み込む
    flat_filename = f"{row[desc_col]}{file_num}_tr.fits"
    flat_path = input_data_dir / flat_filename

    try:
        print(f"  Adding: {flat_path.name}")
        flat_data = fits.getdata(flat_path, ext=0).astype(np.float64)

        if raw_master_flat is None:
            raw_master_flat = flat_data
        else:
            raw_master_flat += flat_data
        led_file_count += 1

    except FileNotFoundError:
        print(f"  警告: フラットファイル {flat_path.name} が見つかりませんでした。スキップします。")

if raw_master_flat is None:
    print("エラー: 有効なLEDファイルが1つも見つからなかったため、処理を終了します。")
    sys.exit()

print(f"-> {led_file_count}個のフラットフレームを合計して、マスターフラットを作成しました。")

# --- 2. 科学データのフラット補正 ---
# 元のパイプラインのロジック（ファイバーごとに割り算→再スケール）を適用
print("\nStep 2: Applying flat-field correction to science frames...")

#science_df = df[df[desc_col] != 'LED']
science_df = df

for index, row in science_df.iterrows():
    stem = Path(row[fits_col]).stem
    file_num = get_file_number(stem)
    if file_num is None:
        print(f"  警告: ファイル名から番号を抽出できませんでした: {stem}")
        continue

    # 科学データの1次元スペクトルファイル (_tr.fits) を読み込む
    science_filename_in = f"{row[desc_col]}{file_num}_tr.fits"
    science_path = input_data_dir / science_filename_in
    print(f"  Processing: {science_path.name}")

    try:
        science_data, header = fits.getdata(science_path, ext=0, header=True)
        science_data = science_data.astype(np.float64)

        # 補正後のデータを格納する空の配列を準備
        corrected_data = np.zeros_like(science_data)

        # --- ファイバーごとにループ処理 ---
        for i in range(science_data.shape[0]):
            science_fiber = science_data[i, :]
            flat_fiber = raw_master_flat[i, :]

            # フラットの中央値を取得（波長範囲512-1536は元のコードに倣う）
            # この範囲は、スペクトルの比較的フラットで安定した部分を選ぶのが一般的
            stable_range = flat_fiber[512:1536]

            # 範囲内に有効なデータがあるか確認
            if stable_range.size > 0:
                median_of_flat_fiber = np.median(stable_range)
            else:
                median_of_flat_fiber = 0

            # ゼロ除算を回避
            if median_of_flat_fiber == 0:
                # 補正係数が0の場合は、補正を適用せず元のデータをコピー
                corrected_data[i, :] = science_fiber
                continue

            # --- ここが核心部分 ---
            # 1. 割り算で感度ムラを補正
            # 2. フラットの中央値を掛けて、元の光量スケールに戻す
            corrected_fiber = np.divide(
                science_fiber * median_of_flat_fiber,
                flat_fiber,
                out=np.zeros_like(science_fiber),
                where=flat_fiber != 0
            )
            corrected_data[i, :] = corrected_fiber

        # 補正後のデータを保存
        output_filename = f"{row[desc_col]}{file_num}_f.fits"
        output_path = output_dir / output_filename
        header['HISTORY'] = 'Flat-field corrected by flat_correction.py'
        fits.writeto(output_path, corrected_data.astype(np.float32), header=header, overwrite=True)
        print(f"  -> Saved corrected file: {output_path.name}")

    except FileNotFoundError:
        print(f"  警告: 科学データ {science_path.name} が見つかりませんでした。スキップします。")

print('\nAll processing completed.')