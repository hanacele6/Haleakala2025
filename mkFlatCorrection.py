import numpy as np
from astropy.io import fits
from pathlib import Path
import os

# --- パス設定 ---
# ご自身の環境に合わせて修正してください
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
fileF1 = base_dir / "output/test"
fileF2 = base_dir / "output/test/"

# --- 処理する科学データのファイル番号 ---
is_num = 10001
ie_num = 10016

# --- 出力ディレクトリの作成 ---
# exist_ok=True は、ディレクトリが既に存在してもエラーにしないオプション
os.makedirs(fileF2, exist_ok=True)

# --- 1. マスターフラットの作成 ---
print('Creating master flat...')

# 基準となるファイルから配列の形状を取得
try:
    with fits.open(fileF1 / f"{is_num}_tr_python.fit") as hdul:
        template_shape = hdul[0].data.shape
except FileNotFoundError:
    print(f"エラー: 基準ファイル {is_num}_tr.fit が見つかりません。")
    exit()

# マスターフラットを倍精度(float64)で初期化
master_flat = np.zeros(template_shape, dtype=np.float64)

# 複数のフラット画像を読み込み、足し合わせる (コンポジット)
for i in range(10013, 10017):   #10013~10016まで
    flat_path = fileF1 / f"{i}_tr_python.fit"
    print(f"  Adding flat frame: {flat_path.name}")
    try:
        # 読み込む際に倍精度(float64)に変換し、一貫性を保つ
        flat_data = fits.getdata(flat_path, ext=0).astype(np.float64)
        master_flat += flat_data
    except FileNotFoundError:
        print(f"  警告: フラットファイル {flat_path.name} が見つかりませんでした。スキップします。")

# --- 2. マスターフラットの規格化（ノーマライズ） ---
# 注意：NumPyのインデックスは [行, 列]、IDLのインデックスは [列, 行] です。
# IDLの flat[642, 72] は、NumPyでは master_flat[72, 642] に相当します。
row_idx, col_idx = 72, 642
norm_value = master_flat[row_idx, col_idx]
print(f"Normalizing master flat with value at (row={row_idx}, col={col_idx}): {norm_value}")

# ゼロ除算を避ける
if norm_value != 0:
    master_flat /= norm_value
else:
    print("警告: 規格化ピクセルの値がゼロです。規格化をスキップします。")

# 完成したマスターフラットを保存
master_flat_path = fileF1 / "flat.fit"
fits.writeto(master_flat_path, master_flat, overwrite=True)
print(f"Master flat saved to: {master_flat_path}")

# --- 3. 科学データのフラット補正 ---
print('Applying flat field correction to science frames...')
for i in range(is_num, ie_num + 1):
    science_path = fileF1 / f"{i}_tr_python.fit"
    print(f"  Processing: {science_path.name}")

    try:
        # 科学データを倍精度で読み込み
        science_data = fits.getdata(science_path, ext=0).astype(np.float64)

        # 割り算を実行して補正
        # np.divideを使い、master_flatが0のピクセルは結果を0にする安全な処理
        corrected_data = np.divide(science_data, master_flat,
                                   out=np.zeros_like(science_data),
                                   where=master_flat != 0)

        # 補正後のデータを保存
        output_path = fileF2 / f"{i}_f_python.fit"
        fits.writeto(output_path, corrected_data, overwrite=True)

    except FileNotFoundError:
        print(f"  警告: 科学データ {science_path.name} が見つかりませんでした。スキップします。")

print('Flat fielding completed.')