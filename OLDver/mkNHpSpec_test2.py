import numpy as np
from astropy.io import fits
from pathlib import Path
import os
from scipy.ndimage import median_filter


# 自動ホットピクセル除去関数（変更なし、このバージョンでは呼び出されない）
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
    IDLコードのロジック（Y方向の上下5ピクセルの平均で補間）を再現する。
    IDLのROI指定が「左下原点Y上向き」であることを前提に、PythonでY座標を変換する。
    """
    if not roi_list:
        return image_data

    print(f"  > 手動でのピクセル補正を開始 ({len(roi_list)} 個の領域)...")
    corrected_data = image_data.copy()  # image_dataはmainでfloat64に変換済み

    # 画像全体の高さを取得（Y軸方向のピクセル数）
    # main関数で data_to_process は [Y, X] の形になっているため、shape[0]がY軸の高さ
    height = image_data.shape[0]  # Y軸の高さ (転置後のデータ配列の高さ)

    for i, roi in enumerate(roi_list):
        # IDLの1ベース座標: [X開始, Y開始, X終了, Y終了] (左下原点, Y上向き)
        x_start_idl, y_start_idl, x_end_idl, y_end_idl = roi

        # Pythonの0ベース座標に変換:
        # X座標はそのまま変換
        ixs = x_start_idl - 1
        ixe = x_end_idl - 1

        # Y座標の変換: IDLの左下原点Y上向き を Pythonの左上原点Y下向き に変換
        # IDLのY_start (矩形の下端) が、Pythonでは (高さ - IDLのY_end) に対応
        # IDLのY_end   (矩形の上端) が、Pythonでは (高さ - IDLのY_start) に対応
        # NumPyのスライスは [start:end) で end は含まれないため、+1 する

        # PythonのYスライス開始インデックス (左上から数えて)
        py_y_slice_start = height - y_end_idl  # IDLのY_end (上端) が Pythonでは下端になる

        # PythonのYスライス終了インデックス (左上から数えて)
        py_y_slice_end = height - y_start_idl + 1  # IDLのY_start (下端) が Pythonでは上端になる

        # 補間値を計算するための上下の領域を定義
        # これらのスライスも、新しいY座標システム (左上原点, Y下向き) に基づく
        y_range_below = slice(py_y_slice_start - 5, py_y_slice_start)
        y_range_above = slice(py_y_slice_end, py_y_slice_end + 5)  # スライスの終端は含まれないため注意

        try:
            mean_below = np.mean(corrected_data[y_range_below, ixs:ixe + 1], axis=0)
            mean_above = np.mean(corrected_data[y_range_above, ixs:ixe + 1], axis=0)
            fill_value = (mean_below + mean_above) / 2.0

            # 補間値を代入する領域も新しいY座標システムで指定
            # 変数名を修正: iye_py -> py_y_slice_end
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
    # ===================================================================
    # --- ユーザー設定 ---

    date = 'test'
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/data") / date
    output_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output") / date

    file_basename = 'mc01'
    file_numbers = [1]

    DATA_FORMAT_FOR_THIS_RUN = 'old'

    MEDIAN_FILTER_KERNEL_SIZE = 3
    SIGMA_THRESHOLD = 10000.0  # 自動除去はほぼ無効

    MANUAL_ROI_LIST = [
    #    [374, 459, 388, 470], [509, 555, 513, 559], [50, 888, 54, 891], [284, 895, 287, 898], [326, 887, 328, 891],
    #    [341, 889, 343, 892], [369, 860, 372, 863], [370, 861, 372, 863],
    ]
    # ===================================================================

    os.makedirs(output_dir, exist_ok=True)
    print(f"--- バッチ処理を開始します ---\n入力ディレクトリ: {base_dir}\n出力ディレクトリ: {output_dir}")
    print(f"データ形式: '{DATA_FORMAT_FOR_THIS_RUN}' (波長方向が横軸か縦軸か)")

    for num in file_numbers:
        filename_base_with_num = f"{file_basename}_{num}"
        input_filename = next(base_dir.glob(f"{filename_base_with_num}.fi*"), None)

        if not input_filename:
            print(
                f"\n[{num}/{len(file_numbers)}] ファイルが見つかりません: {filename_base_with_num}.fi* スキップします。")
            continue

        output_filename = output_dir / (filename_base_with_num + "_nhp_py.fits")
        print(f"\n[{num}/{len(file_numbers)}] 処理中: {input_filename.name}")

        try:
            with fits.open(input_filename) as hdul:
                image_data = hdul[0].data.astype(np.float64)
                header = hdul[0].header
            print(f"  > ファイル読み込み完了。サイズ: {image_data.shape} (縦, 横)")
        except Exception as e:
            print(f"  > FITSファイルの読み込み中にエラー: {e}")
            continue

        data_to_process = image_data.T if DATA_FORMAT_FOR_THIS_RUN == 'old' else image_data

        # --- ステップ1: 自動ホットピクセル除去 (このバージョンではスキップ) ---
        corrected_auto = find_and_correct_hotpixels_auto(
            data_to_process,
            kernel_size=MEDIAN_FILTER_KERNEL_SIZE,
            sigma_threshold=SIGMA_THRESHOLD # 適切な値に調整
        )
        # --- ステップ2: 手動ピクセル補正 ---
        final_corrected = correct_pixels_manual(
            corrected_auto,
            MANUAL_ROI_LIST
        )

        # 最後に元のデータ形式の向きに戻す
        output_image = final_corrected.T if DATA_FORMAT_FOR_THIS_RUN == 'old' else final_corrected

        # --- 補正後のデータを新しいFITSファイルに保存 ---
        try:
            new_hdu = fits.PrimaryHDU(data=output_image, header=header)
            new_hdu.writeto(output_filename, overwrite=True)
            print(f"  > 処理完了。結果を '{output_filename.name}' に保存しました。")
        except Exception as e:
            print(f"  > FITSファイルの保存中にエラー: {e}")

    print("\n--- 全ての処理が完了しました ---")


if __name__ == '__main__':
    main()