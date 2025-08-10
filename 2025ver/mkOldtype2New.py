import os
import glob
from astropy.io import fits
import numpy as np
from scipy.ndimage import zoom


def process_fits(input_path: str, output_path: str, new_height: int, new_width: int, rotation: str = 'right'):
    """
    FITSファイルをリサイズし、回転させる関数。（この関数は変更ありません）

    Args:
        input_path (str): 入力FITSファイルのパス。
        output_path (str): 出力FITSファイルのパス。
        new_height (int): リサイズ後の高さピクセル数。
        new_width (int): リサイズ後の幅ピクセル数。
        rotation (str): 回転方向。'right' (時計回り) または 'left' (反時計回り) を指定。
    """
    # 1. FITSファイルの読み込み
    with fits.open(input_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    original_height, original_width = data.shape
    print(f"'{os.path.basename(input_path)}' を読み込みました。サイズ: {original_width}x{original_height}")

    # 2. リサイズ (補間)
    zoom_factor_h = new_height / original_height
    zoom_factor_w = new_width / original_width
    resized_data = zoom(data, (zoom_factor_h, zoom_factor_w), order=3)

    # 3. 回転
    if rotation == 'right':
        rotated_data = np.rot90(resized_data, k=-1)
    elif rotation == 'left':
        rotated_data = np.rot90(resized_data, k=1)
    elif rotation == 'none':
        rotated_data = resized_data
    else:
        raise ValueError("回転方向は 'right' または 'left' で指定してください。")

    # 4. ヘッダーの更新とFITSファイルの保存
    new_header = header.copy()
    new_header['NAXIS1'] = rotated_data.shape[1]
    new_header['NAXIS2'] = rotated_data.shape[0]
    new_header.add_history(f"Resized from {original_width}x{original_height} to {new_width}x{new_height}.")
    new_header.add_history(f"Rotated 90 degrees to the {rotation}.")

    hdu_new = fits.PrimaryHDU(rotated_data, header=new_header)
    hdul_new = fits.HDUList([hdu_new])
    hdul_new.writeto(output_path, overwrite=True)

    print(f"✅ 処理完了 → '{output_path}'")


# --- 実行セクション (ディレクトリ内の全ファイルを処理) ---
if __name__ == '__main__':
    # --- パラメータを設定してください ---

    # 処理対象のFITSファイルが保存されているディレクトリ ('.': このスクリプトと同じディレクトリ)
    input_directory = '.'

    # 処理後のファイルを保存するディレクトリ名
    output_directory = 'processed_files'

    # 目標とするサイズ（リサイズ後、回転前）
    target_height = 2048
    target_width = 1024

    # 回転方向: 'right' (右・時計回り) または 'left' (左・反時計回り)
    rotation_direction = 'right'

    # ------------------------------------

    # 出力ディレクトリが存在しない場合は作成する
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"出力ディレクトリ '{output_directory}' を作成しました。")

    # FITSの一般的な拡張子をリストアップ
    extensions = ['*.fits', '*.fit', '*.fts']
    file_list = []
    for ext in extensions:
        file_list.extend(glob.glob(os.path.join(input_directory, ext)))

    # 処理対象ファイルがない場合はメッセージを表示
    if not file_list:
        print(f"処理対象のFITSファイルが '{input_directory}' ディレクトリに見つかりませんでした。")
    else:
        print(f"Found {len(file_list)} FITS file(s). Starting processing...")

        # 取得したファイルリストをループして一つずつ処理
        for input_file_path in file_list:
            # 出力ファイル名を生成 (例: my_image.fits -> processed_files/my_image_processed.fits)
            base_name = os.path.basename(input_file_path)
            name, ext = os.path.splitext(base_name)
            output_filename = f"{name}_processed{ext}"
            output_file_path = os.path.join(output_directory, output_filename)

            print("-" * 50)

            try:
                # 各ファイルに対して処理関数を実行
                process_fits(
                    input_path=input_file_path,
                    output_path=output_file_path,
                    new_height=target_height,
                    new_width=target_width,
                    rotation=rotation_direction
                )
            except Exception as e:
                print(f"❌ エラー: '{input_file_path}' の処理中に問題が発生しました。")
                print(f"   詳細: {e}")

    print("\nすべての処理が完了しました。")