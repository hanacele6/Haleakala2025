import numpy as np
from astropy.io import fits
from pathlib import Path
import os
from scipy.ndimage import median_filter  # メディアンフィルタを使うためにインポート


def find_and_correct_hotpixels_auto(image_data, kernel_size=3, sigma_threshold=5.0):
    """
    統計的な手法を用いて、単一の画像フレーム内のホットピクセルを自動で検出し補正する。

    Args:
        image_data (np.ndarray): 補正対象の2次元画像データ。
        kernel_size (int): ホットピクセル検出に使うメディアンフィルタのサイズ。
        sigma_threshold (float): ホットピクセルと判断する、標準偏差の何倍かの閾値。

    Returns:
        np.ndarray: 補正後の画像データ。
    """
    print(f"  > 自動ホットピクセル除去を開始 (kernel_size={kernel_size}, sigma_threshold={sigma_threshold})...")

    # copy() を使って、元の画像データを変更しないようにする
    corrected_data = image_data.copy()

    # 1. メディアンフィルタを適用して、滑らかな「お手本」画像を作成
    smoothed_image = median_filter(corrected_data, size=kernel_size)

    # 2. 元画像と平滑化画像の差分を計算
    difference_map = corrected_data - smoothed_image

    # 3. 差分画像の統計値からホットピクセルを特定
    # 上位1%の極端な値を除外して、より信頼性の高い標準偏差を計算する
    threshold = np.percentile(difference_map, 99)
    std_dev = np.std(difference_map[difference_map < threshold])

    # 差分が「(クリップ後の)標準偏差のN倍」を超える点をホットピクセルと判断
    hot_pixel_threshold = sigma_threshold * std_dev

    # デバッグ用に標準偏差の値を表示
    print(f"  > 差分画像の標準偏差(上位1%除外): {std_dev:.4f}, 判定閾値: {hot_pixel_threshold:.4f}")

    # ホットピクセルの場所を示すマスク（True/Falseの配列）を作成
    hot_pixels_mask = difference_map > hot_pixel_threshold

    num_hot_pixels = np.sum(hot_pixels_mask)
    if num_hot_pixels > 0:
        print(f"  > {num_hot_pixels} 個のホットピクセルを検出しました。補正します...")
        # 4. ホットピクセルを、平滑化画像の対応するピクセルの値で置き換える
        corrected_data[hot_pixels_mask] = smoothed_image[hot_pixels_mask]
    else:
        print("  > ホットピクセルは検出されませんでした。")

    return corrected_data


def correct_pixels_manual(image_data, roi_list):
    """
    ユーザーが指定したROI（関心領域）リストに基づき、ピクセルを手動で補正する。
    IDLコードのロジック（Y方向の上下5ピクセルの平均で補間）を再現する。

    Args:
        image_data (np.ndarray): 補正対象の2次元画像データ。
        roi_list (list): [[x_start, y_start, x_end, y_end], ...] の形式のROIリスト。

    Returns:
        np.ndarray: 補正後の画像データ。
    """
    if not roi_list:
        return image_data

    print(f"  > 手動でのピクセル補正を開始 ({len(roi_list)} 個の領域)...")
    corrected_data = image_data.copy()

    for i, roi in enumerate(roi_list):
        # IDLの1ベース座標をPythonの0ベースインデックスに変換
        x_start, y_start, x_end, y_end = roi
        ixs, iys = x_start - 1, y_start - 1
        ixe, iye = x_end - 1, y_end - 1

        # 補間値を計算するための上下の領域を定義 (Y方向の上下5ピクセル)
        y_range_below = slice(iys - 5, iys)
        y_range_above = slice(iye + 1, iye + 6)

        try:
            # NumPyを使い、X方向のforループをなくして高速化
            mean_below = np.mean(corrected_data[y_range_below, ixs:ixe + 1], axis=0)
            mean_above = np.mean(corrected_data[y_range_above, ixs:ixe + 1], axis=0)
            fill_value = (mean_below + mean_above) / 2.0
            corrected_data[iys:iye + 1, ixs:ixe + 1] = fill_value
        except (IndexError, ValueError) as e:
            print(
                f"    > 警告: ROI #{i + 1} ([{x_start}, {y_start}, ...]) が画像の端に近すぎるなどの理由で補間できません。スキップします。({e})")
            continue

    return corrected_data


def main():
    """
    メインの処理フロー
    """
    # ===================================================================
    # --- ユーザー設定 ---

    # 1. 基本となるディレクトリのパス

    date = 'test'  # ← ここを編集 (必要なら)
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/data") / date
    output_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output") / date

    # 2. 処理したいファイル名の共通部分と連番

    file_basename = 'mc01'  # ← ここを編集
    file_numbers = [1]  # ← ここを編集

    # 3. データの向き ('old' または 'new' を指定)
    # 'old': 縦軸が波長方向
    # 'new': 横軸が波長方向
    DATA_FORMAT_FOR_THIS_RUN = 'new'

    # 4. 自動除去のパラメータ
    MEDIAN_FILTER_KERNEL_SIZE = 3  # 通常は3でOK
    SIGMA_THRESHOLD = 500.0  # 5.0が安全な初期値。結果に応じて調整。

    # 5. 手動で補正するROIのリスト ([X開始, Y開始, X終了, Y終了])
    # 手動補正が不要な場合は、リストを空にします (MANUAL_ROI_LIST = [])
    MANUAL_ROI_LIST = [
        #[12, 2782, 19, 2790],  # 例: IDLコードの最初のROI
        # [564, 2742, 574, 2751], # 例: IDLコードの2番目のROI
        #[374, 470, 388, 459],
        #[374, 459, 388, 470],
        #[388, 470, 374, 459],
        #[388, 459, 374, 470],
        #[459,374, 470, 388],
        #[470, 374, 459, 388],
        #[459, 388, 470, 374],
        #[470, 374, 459, 388],

    ]
    # ===================================================================

    os.makedirs(output_dir, exist_ok=True)
    print(f"--- バッチ処理を開始します ---\n入力ディレクトリ: {base_dir}\n出力ディレクトリ: {output_dir}")
    print(f"データ形式: '{DATA_FORMAT_FOR_THIS_RUN}' (波長方向が横軸か縦軸か)")

    for num in file_numbers:
        filename_base_with_num = f"{file_basename}_{num}"
        input_filename = next(base_dir.glob(f"{filename_base_with_num}.fi*"), None)  # .fit と .fits の両対応

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

        # データの向きを、内部処理用に[ファイバー, 波長]に統一する
        # 'old'は[波長, ファイバー]なので転置が必要
        data_to_process = image_data.T if DATA_FORMAT_FOR_THIS_RUN == 'old' else image_data

        # --- ステップ1: 自動ホットピクセル除去 ---
        corrected_auto = find_and_correct_hotpixels_auto(
            data_to_process,
            kernel_size=MEDIAN_FILTER_KERNEL_SIZE,
            sigma_threshold=SIGMA_THRESHOLD
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