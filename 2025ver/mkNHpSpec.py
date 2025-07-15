import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import os
from scipy.ndimage import median_filter

# --- ユーザー設定 ---

# 処理結果を保存するディレクトリ
date = "20250501"
output_path = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/")
output_dir = output_path / date

# 処理対象のファイルリストが書かれたCSVファイル
CSV_FILE = "mcparams202505.csv"

# ホットピクセル除去のパラメータ
MEDIAN_FILTER_KERNEL_SIZE = 3
SIGMA_THRESHOLD = 3000.0  # 自動除去のしきい値

# 手動で補正したい領域があれば、ここのコメントを外して指定
MANUAL_ROI_LIST = [
    # [374, 459, 388, 470], [509, 555, 513, 559],
]


# --- 関数定義 (変更なし) ---

def find_and_correct_hotpixels_auto(image_data, kernel_size=3, sigma_threshold=5.0):
    """
    統計的な手法を用いて、単一の画像フレーム内のホットピクセルを自動で検出し補正する。
    """
    print(f"  > 自動ホットピクセル除去を開始 (kernel_size={kernel_size}, sigma_threshold={sigma_threshold})...")
    corrected_data = image_data.copy()
    smoothed_image = median_filter(corrected_data, size=kernel_size)
    difference_map = corrected_data - smoothed_image
    threshold = np.percentile(difference_map, 99)
    std_dev = np.std(difference_map[difference_map < threshold])
    hot_pixel_threshold = sigma_threshold * std_dev
    print(f"  > 差分画像の標準偏差(上位1%除外): {std_dev:.4f}, 判定閾値: {hot_pixel_threshold:.4f}")
    hot_pixels_mask = difference_map > hot_pixel_threshold
    num_hot_pixels = np.sum(hot_pixels_mask)
    if num_hot_pixels > 0:
        print(f"  > {num_hot_pixels} 個のホットピクセルを検出しました。補正します...")
        corrected_data[hot_pixels_mask] = smoothed_image[hot_pixels_mask]
    else:
        print("  > ホットピクセルは検出されませんでした。")
    return corrected_data


def correct_pixels_manual(image_data, roi_list):
    """
    ユーザーが指定したROI（関心領域）リストに基づき、ピクセルを手動で補正する。
    IDLのROI指定が「左下原点Y上向き」であることを前提に、PythonでY座標を変換する。
    """
    if not roi_list:
        return image_data

    print(f"  > 手動でのピクセル補正を開始 ({len(roi_list)} 個の領域)...")
    corrected_data = image_data.copy()
    height = image_data.shape[0]

    for i, roi in enumerate(roi_list):
        x_start_idl, y_start_idl, x_end_idl, y_end_idl = roi
        ixs = x_start_idl - 1
        ixe = x_end_idl - 1
        py_y_slice_start = height - y_end_idl
        py_y_slice_end = height - y_start_idl + 1

        y_range_below = slice(py_y_slice_start - 5, py_y_slice_start)
        y_range_above = slice(py_y_slice_end, py_y_slice_end + 5)

        try:
            mean_below = np.mean(corrected_data[y_range_below, ixs:ixe + 1], axis=0)
            mean_above = np.mean(corrected_data[y_range_above, ixs:ixe + 1], axis=0)
            fill_value = (mean_below + mean_above) / 2.0
            corrected_data[py_y_slice_start:py_y_slice_end, ixs:ixe + 1] = fill_value
        except (IndexError, ValueError) as e:
            print(
                f"    > 警告: ROI #{i + 1} ([{x_start_idl}, {y_start_idl}, ...]) が画像の端に近すぎるなどの理由で補間できません。スキップします。({e})")
            continue

    return corrected_data


def main():
    """
    メインの処理フロー
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"--- バッチ処理を開始します ---\n出力ディレクトリ: {output_dir}")

    # --- CSVファイルから処理対象リストを読み込む ---
    if not os.path.exists(CSV_FILE):
        print(f"エラー: CSVファイルが見つかりません: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    if df.empty or len(df.columns) == 0:
        print(f"エラー: CSVファイルが空か、列がありません。")
        return

    # CSVの1列目をFITSファイルのパスリストとして取得
    fits_col_name = df.columns[0]
    file_list = df[fits_col_name].tolist()

    print(f"情報: CSVファイル '{CSV_FILE}' の '{fits_col_name}' 列から {len(file_list)} 件のファイルを処理します。")

    # --- 読み込んだリストのファイルを一つずつ処理 ---
    for i, filepath_str in enumerate(file_list):
        input_filename = Path(filepath_str)

        if not input_filename.is_file():
            print(f"\n[{i + 1}/{len(file_list)}] ファイルが見つかりません: {input_filename} スキップします。")
            continue

        # 出力ファイル名を生成 (例: mc01_1.fits -> mc01_1_nhp_py.fits)
        output_filename = output_dir / (input_filename.stem + "_nhp_py.fits")
        print(f"\n[{i + 1}/{len(file_list)}] 処理中: {input_filename.name}")

        try:
            with fits.open(input_filename) as hdul:
                image_data = hdul[0].data.astype(np.float64)
                header = hdul[0].header
            print(f"  > ファイル読み込み完了。サイズ: {image_data.shape} (縦, 横)")
        except Exception as e:
            print(f"  > FITSファイルの読み込み中にエラー: {e}")
            continue

        # 'new'形式を前提とするため、データはそのまま使用
        data_to_process = image_data

        # ステップ1: 自動ホットピクセル除去
        corrected_auto = find_and_correct_hotpixels_auto(
            data_to_process,
            kernel_size=MEDIAN_FILTER_KERNEL_SIZE,
            sigma_threshold=SIGMA_THRESHOLD
        )

        # ステップ2: 手動ピクセル補正
        final_corrected = correct_pixels_manual(
            corrected_auto,
            MANUAL_ROI_LIST
        )

        output_image = final_corrected

        # 補正後のデータを新しいFITSファイルに保存
        try:
            new_hdu = fits.PrimaryHDU(data=output_image, header=header)
            new_hdu.writeto(output_filename, overwrite=True)
            print(f"  > 処理完了。結果を '{output_filename.name}' に保存しました。")
        except Exception as e:
            print(f"  > FITSファイルの保存中にエラー: {e}")

    print("\n--- 全ての処理が完了しました ---")


if __name__ == '__main__':
    main()