import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import sys

# ==============================================================================
# 設定項目 (★ここを自分の環境に合わせて変更してください)
# ==============================================================================

# 1. 解析したいFITSファイルのパスを指定
#    夕方の観測で得られたマスターskyフレームなどを指定します。
fits_filepath = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20251017/master_sky.fits" # ★要変更
#fits_filepath = "C:/Users/hanac/University/Senior/PythonProject/merc2025a/fits/20250501/sky01r_sp.wmp.fits"
#fits_filepath = "C:/Users/hanac/University/Senior/Mercury/Haleakala2025/output/20150223/MERCURY1_tr.wc.fits" # ★要変更
# 2. どのファイバーのスペクトルを見るか指定
#    画像の真ん中あたりのファイバー番号が良いでしょう（例：50、100など）
fiber_to_inspect = 1  # ★お好みで変更

# 3. 出力するdatファイルのパス
#    指定しない場合は、入力FITSファイルと同じ場所に「(入力ファイル名)_fiber(番号).dat」という名前で保存されます。
output_dat_filepath = "" # (任意)

# ==============================================================================
# メイン処理
# ==============================================================================
if __name__ == "__main__":
    # --- 入力ファイルの存在確認 ---
    if not os.path.exists(fits_filepath):
        print(f"エラー: 指定されたFITSファイルが見つかりません。")
        print(f"ファイルパス: {fits_filepath}")
        sys.exit()

    print(f"FITSファイルを読み込んでいます: {fits_filepath}")

    # --- FITSファイルからデータを読み込み ---
    try:
        with fits.open(fits_filepath) as hdul:
            data = hdul[0].data
            ny, nx = data.shape  # ny: ファイバー数, nx: ピクセル数

            # 指定されたファイバー番号が妥当かチェック
            if fiber_to_inspect >= ny:
                print(f"エラー: 指定されたファイバー番号 ({fiber_to_inspect}) は、画像のファイバー数 ({ny}) を超えています。")
                sys.exit()

            # 指定されたファイバーのスペクトルデータを1次元配列として抽出
            spectrum = data[fiber_to_inspect, :]
            pixels = np.arange(nx)
            print(f"ファイバー番号 {fiber_to_inspect} のデータを抽出しました。")

    except Exception as e:
        print(f"FITSファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit()

    # --- datファイルとして保存 ---
    if not output_dat_filepath:
        base_name = os.path.splitext(os.path.basename(fits_filepath))[0]
        output_dat_filepath = os.path.join(os.path.dirname(fits_filepath), f"{base_name}_fiber{fiber_to_inspect}.dat")

    # データを2列（ピクセル, 強度）に整形して保存
    output_data = np.column_stack((pixels, spectrum))
    header_text = f"Spectrum data for fiber {fiber_to_inspect} from {os.path.basename(fits_filepath)}\nPixel_X  Intensity"
    np.savetxt(output_dat_filepath, output_data, fmt='%d  %.4f', header=header_text)
    print(f"スペクトルデータをテキストファイルとして保存しました: {output_dat_filepath}")


    # --- スペクトルをプロットして表示 ---
    print("スペクトルをプロットします。ウィンドウを拡大・操作してピクセル位置を確認してください。")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(pixels, spectrum, label=f"Fiber {fiber_to_inspect}")
    ax.set_title(f"Spectrum from Fiber {fiber_to_inspect} of {os.path.basename(fits_filepath)}", fontsize=16)
    ax.set_xlabel("Pixel (Dispersion Axis)", fontsize=12)
    ax.set_ylabel("Intensity (ADU)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.show()