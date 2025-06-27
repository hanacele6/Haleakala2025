import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
import os
import time

# --- 設定値 (Setting Values) ---
# ご自身の環境に合わせてパスやファイル名を変更してください
date = '20250501'
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
# 出力用ディレクトリ
output_dir = base_dir / f"output/{date}"
# データ用ディレクトリ
data_dir = base_dir / f"data/{date}"

# ディレクトリが存在しない場合に作成
output_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# --- FITSファイルの読み込み (Reading FITS files) ---
try:
    # fppolyのファイル名はご自身のものに合わせてください
    with fits.open(output_dir / 'fppoly1.fits') as hdul:
        # fppolyは (ファイバー数, 波長) の次元と仮定
        fibp = hdul[0].data
except FileNotFoundError as e:
    print(f"エラー: 必要なFITSファイルが見つかりません: {e}")
    print("fppoly1.fitが output/{date} ディレクトリに存在することを確認してください。")
    exit()

# fibpの形状をデバッグ表示
print(f"DEBUG: Shape of fibp (from fppoly1.fit): {fibp.shape}")

# --- ファイルリストを読み込む (Reading file list) ---
list_txt_path = output_dir / 'list.txt'
file_list = []
try:
    with open(list_txt_path, 'r') as f:
        # 各行の末尾にある改行文字などの空白を削除してリストに格納
        file_list = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"エラー: ファイルリスト '{list_txt_path}' が見つかりません。プログラムを終了します。")
    exit()

ifilem = len(file_list)

if not file_list:
    print("ファイルリストが空です。処理を終了します。")
    exit()

print(f"'{list_txt_path}' から {ifilem} 個のファイルを読み込みました。")
print(file_list)

# --- 定数設定 (Constants Definition) ---
# 2025年バージョンに合わせた定数
NX = 2048  # 波長方向のピクセル数 (Number of pixels in wavelength direction)
NY = 1024  # 空間方向のピクセル数 (Number of pixels in spatial direction)
ifibm = fibp.shape[0] if fibp.ndim > 1 else 1  # ファイバー数 (fiber number)
hwid = 2  # ガウスフィットの半幅 (half-width for Gaussian fit)

print(f"DEBUG: NX={NX}, NY={NY}, ifibm={ifibm}")


# --- Gauss関数定義 (Gaussian function for curve_fit) ---
def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + offset


# --- poserr.txt ファイルの書き込み準備 ---
poserr_path = output_dir / 'poserr_python.txt'
with open(poserr_path, 'w') as lunw2:
    total_start_time = time.time()  # 全体の処理開始時間

    # --- ファイルごとの処理ループ ---
    for ifile, fileid in enumerate(file_list):
        file_start_time = time.time()  # ファイルごとの処理開始時間
        print(f"\nファイル {ifile + 1}/{ifilem} を処理中: {fileid}")

        # --- データFITSファイルの読み込み ---
        data_file_path = output_dir / fileid
        if not data_file_path.exists():
            print(f"警告: データファイル {data_file_path} が見つかりませんでした。スキップします。")
            continue

        with fits.open(data_file_path) as hdul:
            # bの形状は (NY, NX) = (空間, 波長) と仮定
            b = hdul[0].data.astype(np.float64)

        # 配列の初期化
        fiblall2 = np.zeros((ifibm, NX), dtype=np.float64)

        # --- ファイバーごとの処理ループ ---
        for ifib in range(ifibm):
            fib_start_time = time.time()
            # print(f"  ファイバー {ifib + 1}/{ifibm} を処理中...")

            # 1本のファイバーの軌跡（空間方向のピクセル位置）を抜き出す
            fibpix = fibp[ifib, :]

            # ファイバーの軌道データにNaNが含まれているかチェック
            if np.isnan(fibpix).any():
                print(f"  警告: ファイバー {ifib} はNaN値を含むためスキップします。")
                fiblall2[ifib, :] = 0.0  # 出力値を0で埋める
                continue  # 次のファイバーへ

            # 配列の初期化
            fibl = np.zeros((hwid * 2 + 1, NX), dtype=np.float64)

            # --- 波長ごとの処理ループ ---
            for ix in range(NX):
                y_pixel_indices = np.arange(NY)
                spatial_data_slice = b[:, ix]
                interp_func = interp1d(
                    y_pixel_indices,
                    spatial_data_slice,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(spatial_data_slice[0], spatial_data_slice[-1])
                )
                for iy in range(-hwid, hwid + 1):
                    pos_to_interp = fibpix[ix] + iy
                    fibl[iy + hwid, ix] = interp_func(pos_to_interp)

            # --- ガウスフィットによる中心位置の精密化 ---
            fibl2 = np.sum(fibl, axis=1)

            x_fit = np.arange(hwid * 2 + 1)
            initial_guess = [np.max(fibl2) - np.min(fibl2), float(hwid), 1.0, np.min(fibl2)]

            try:
                params, covariance = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess, method='lm', maxfev=10000)
                arr = params
            except RuntimeError as e:
                print(
                    f"  警告: ガウスフィットが失敗しました (ファイル {ifile}, ファイバー {ifib}): {e}. デフォルト値を使用します。")
                arr = initial_guess

            if abs(arr[1] - hwid) > 1.5:
                lunw2.write(f"{ifile} {ifib} {arr[1]:.6f}\n")
                arr[1] = float(hwid)

            # --- 精密化された中心位置を使って再度スペクトルを抽出 ---
            fibl3 = np.zeros_like(fibl)
            for ix in range(NX):
                y_data_fibl3 = fibl[:, ix]
                x_interp_data = np.arange(fibl.shape[0])

                # fill_valueをextrapolateから、データの両端の値を使うように変更
                interp_func_fibl3 = interp1d(
                    x_interp_data,
                    y_data_fibl3,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(y_data_fibl3[0], y_data_fibl3[-1])
                )
                for iy in range(-hwid, hwid + 1):
                    fibl3[iy + hwid, ix] = interp_func_fibl3(arr[1] + iy)




            fibl4 = np.sum(fibl3[hwid - 1: hwid + 2, :], axis=0) - (fibl3[0, :] + fibl3[hwid * 2, :]) / 2.0 * 3.0

            fiblall2[ifib, :] = fibl4

            if ifib == 45 and ifile == 0:
                save_path = output_dir / 'debug_fibl3_python_fib45.fit'
                print(f'  DEBUG: Saving fibl3 for fiber 45 to {save_path}')
                hdu = fits.PrimaryHDU(fibl3.T)
                hdu.writeto(save_path, overwrite=True)

        # --- FITSファイルへの書き出し ---
        output_file_name = f'{10000 + ifile + 1:05d}_tr_python.fit'
        output_path = output_dir / output_file_name
        hdu = fits.PrimaryHDU(fiblall2)
        hdu.writeto(output_path, overwrite=True)

        file_end_time = time.time()
        print(
            f"ファイル {ifile + 1}/{ifilem} 処理完了。 '{output_file_name}' として保存。所要時間: {file_end_time - file_start_time:.2f}秒")

    total_end_time = time.time()
    print(f"\nすべての処理が完了しました。全体所要時間: {total_end_time - total_start_time:.2f}秒")
