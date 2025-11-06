import numpy as np
from astropy.io import fits
from pathlib import Path
import os
from scipy.ndimage import median_filter  # メディアンフィルタを使うためにインポート


def find_and_correct_hotpixels_auto(image_data, kernel_size=3, sigma_threshold=5.0):
    """
    統計的な手法を用いて、単一の画像フレーム内のホットピクセルを自動で検出し補正する。
    このバージョンでは、デバッグのため、常に元のデータを変更せずに返す。
    """
    print(f"  > 自動ホットピクセル除去はデバッグのためスキップされます。")
    return image_data.copy()


def correct_pixels_manual(image_data, roi_list):
    """
    ユーザーが指定したROI（関心領域）リストに基づき、ピクセルを手動で補正する。
    IDLコードのロジック（Y方向の上下5ピクセルの平均で補間）を再現する。
    IDLが計算結果を最終的に整数に丸めている挙動をPythonで再現する。
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
            # 計算はfloat64で行われる
            mean_below = np.mean(corrected_data[y_range_below, ixs:ixe + 1], axis=0)
            mean_above = np.mean(corrected_data[y_range_above, ixs:ixe + 1], axis=0)

            # 生の補間計算結果 (float64)
            fill_value_raw_float64 = (mean_below + mean_above) / 2.0

            # --- IDLが補間結果を整数に丸めている挙動を再現 ---
            # IDLが1917.9を1917.0にしていることから、np.round()で最も近い整数に丸める
            # そして、その整数値をfloat64の形式で保持する
            #fill_value_final = np.round(fill_value_raw_float64).astype(np.float64)
            fill_value_final = np.floor(fill_value_raw_float64).astype(np.float64)

            # 補間値を代入する
            corrected_data[iys:iye + 1, ixs:ixe + 1] = fill_value_final
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

    date = 'test'
    base_dir = Path("C:/Users/hanac/University/Senior/Mercury/data") / date
    output_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output") / date

    file_basename = 'mc01'
    file_numbers = [1]

    # ここは 'new' のままとします。
    # Pythonの読み込みで転置を行わない（[Y, X]順）が、IDLのデータ配置と合致するという仮説。
    DATA_FORMAT_FOR_THIS_RUN = 'new'

    MEDIAN_FILTER_KERNEL_SIZE = 3
    SIGMA_THRESHOLD = 10000.0

    # テスト対象のROIに絞る
    MANUAL_ROI_LIST = [
          # IDLのY座標は 459 (下) から 470 (上)
        # 他のROIはここでは含めない（テストのため）
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
        print(f"  > ファイル '{input_filename.name}' を処理中...")

        try:
            with fits.open(input_filename) as hdul:
                # IDLの最終出力がBITPIX=-64（float64）だったため、
                # Pythonもfloat64に統一して読み込むのが最も安全
                image_data = hdul[0].data.astype(np.float64)
                header = hdul[0].header
            print(f"  > ファイル読み込み完了。サイズ: {image_data.shape} (縦, 横)")
        except Exception as e:
            print(f"  > FITSファイルの読み込み中にエラー: {e}")
            continue

        data_to_process = image_data.T if DATA_FORMAT_FOR_THIS_RUN == 'old' else image_data

        # --- ステップ1: 自動ホットピクセル除去 ---
        # find_and_correct_hotpixels_auto関数が常に元のデータを返すように修正済み
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