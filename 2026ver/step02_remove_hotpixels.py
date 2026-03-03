import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
import os
from scipy.ndimage import median_filter


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


def run(run_info, config):
    """
    メインの処理フロー (統合パイプラインから呼び出される)
    """
    # 統合スクリプトから自動生成されたパスを受け取る
    output_dir = run_info["output_dir"]
    csv_file = run_info["csv_path"]

    # config.yaml から設定値を安全に読み込む（設定がない場合のデフォルト値も指定）
    kernel_size = config.get("hotpixel", {}).get("kernel_size", 3)
    sigma_threshold = config.get("hotpixel", {}).get("sigma_threshold", 1000.0)
    manual_roi_list = config.get("hotpixel", {}).get("manual_roi", [])
    force_rerun = config.get("pipeline", {}).get("force_rerun_hotpixel", False)

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- ホットピクセル除去処理を開始します ---")
    print(f"出力ディレクトリ: {output_dir}")

    # --- CSVファイルから処理対象リストを読み込む ---
    if not os.path.exists(csv_file):
        print(f"エラー: CSVファイルが見つかりません: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    if df.empty or len(df.columns) == 0:
        print(f"エラー: CSVファイルが空か、列がありません。")
        return

    # CSVの1列目をFITSファイルのパスリストとして取得
    fits_col_name = df.columns[0]
    file_list = df[fits_col_name].tolist()

    print(f"情報: '{csv_file.name}' から {len(file_list)} 件のファイルを処理します。")

    # --- 読み込んだリストのファイルを一つずつ処理 ---
    for i, filepath_str in enumerate(file_list):
        input_filename = Path(filepath_str)

        if not input_filename.is_file():
            print(f"[{i + 1}/{len(file_list)}] ファイルが見つかりません: {input_filename.name} スキップします。")
            continue

        # 出力ファイル名を生成 (例: mc01_1.fits -> mc01_1_nhp_py.fits)
        output_filename = output_dir / (input_filename.stem + "_nhp_py.fits")

        # 出力ファイルがすでに整理フォルダ(1_fits)にあるかどうかもチェック
        output_filename_organized = output_dir / "1_fits" / output_filename.name

        print(f"DEBUG: Check1 (Direct): {output_filename} -> {output_filename.exists()}")
        print(f"DEBUG: Check2 (Organized): {output_filename_organized} -> {output_filename_organized.exists()}")

        is_processed = output_filename.exists() or output_filename_organized.exists()

        # ▼▼▼ 極限まで実行を減らすスキップ処理 ▼▼▼
        if is_processed and not force_rerun:
            print(f"[{i + 1}/{len(file_list)}] 処理済みスキップ: {output_filename.name}")
            continue

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
            kernel_size=kernel_size,
            sigma_threshold=sigma_threshold
        )

        # ステップ2: 手動ピクセル補正
        final_corrected = correct_pixels_manual(
            corrected_auto,
            manual_roi_list
        )

        # output_image = final_corrected
        output_image = np.fliplr(final_corrected)

        # 補正後のデータを新しいFITSファイルに保存 (直下に保存。後で整理される)
        try:
            new_hdu = fits.PrimaryHDU(data=output_image, header=header)
            new_hdu.writeto(output_filename, overwrite=True)
            print(f"  > 処理完了。結果を '{output_filename.name}' に保存しました。")
        except Exception as e:
            print(f"  > FITSファイルの保存中にエラー: {e}")

    print("--- ホットピクセル除去処理が完了しました ---")


# 単体テスト用（必要に応じて）
if __name__ == '__main__':
    print("このスクリプトは main.py からモジュールとして呼び出してください。")