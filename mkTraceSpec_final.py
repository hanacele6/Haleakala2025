import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
import os
import time

# --- 設定値 (IDLコードの変数に対応) ---
date = 'test'
fileF = Path("C:/Users/hanac/University/Senior/Mercury/")
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
fileF1 = base_dir / f"output/{date}"
fileFd = fileF / f"data/{date}"

# ディレクトリが存在しない場合に作成
fileF1.mkdir(parents=True, exist_ok=True)
fileFd.mkdir(parents=True, exist_ok=True)

# --- FITSファイルの読み込み ---
try:
    with fits.open(fileF1 / 'flat1.fit') as hdul:
        flat = hdul[0].data
    with fits.open(fileF1 / 'fppoly1.fit') as hdul:
        fibp = hdul[0].data
except FileNotFoundError as e:
    print(f"エラー: 必要なFITSファイルが見つかりません: {e}")
    exit()

# fibpの形状をデバッグ表示
print(f"DEBUG: Shape of fibp (from fppoly1.fit): {fibp.shape}")

# --- list.txtからファイルリストを読み込む ---
list_txt_path = fileF1 / 'list.txt'
file = []
try:
    with open(list_txt_path, 'r') as f:
        # 各行の末尾にある改行文字などの空白を削除してリストに格納
        file = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"エラー: ファイルリスト '{list_txt_path}' が見つかりません。プログラムを終了します。")
    exit()

ifilem = len(file)

if not file:
    print("ファイルリストが空です。処理を終了します。")
    exit()

print(f"'{list_txt_path}' から {ifilem} 個のファイルを読み込みました。")
print(file)


# --- 定数設定 ---
ifp1 = 0
ifp2 = 35
iym = 1327
ifibm = 93  # fiber number
iys2 = 1 - 1
iye2 = 1327 - 1
iym2 = iye2 - iys2 + 1
hwid = 5


# --- Gauss関数定義 (curve_fit用) ---
def gaussian(x, amplitude, mean, stddev, offset):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + offset


# --- poserr.txt ファイルの書き込み準備 ---
poserr_path = fileF1 / 'poserr_python.txt'
with open(poserr_path, 'w') as lunw2:
    total_start_time = time.time()  # 全体の処理開始時間

    for ifile in range(ifilem):
        fileid = file[ifile]
        file_start_time = time.time()  # ファイルごとの処理開始時間
        print(f"\nファイル {ifile + 1}/{ifilem} を処理中: {fileid}")  # ファイルごとの進捗表示

        # --- FITSデータの読み込み ---
        data_file_path = fileFd / fileid
        if not data_file_path.exists():
            print(f"警告: データファイル {data_file_path} が見つかりませんでした。スキップします。")
            continue

        with fits.open(data_file_path) as hdul:
            b = hdul[0].data

        # 配列の初期化
        fibl = np.zeros((hwid * 2 + 1, iym2), dtype=np.float64)
        fibl2 = np.zeros(hwid * 2 + 1, dtype=np.float64)
        fibl3 = np.zeros_like(fibl)
        fibl4 = np.zeros(iym2, dtype=np.float64)
        fibl5 = np.zeros_like(fibl3)

        fiblall = np.zeros(((hwid * 2 + 1) * ifibm, iym2), dtype=np.float64)
        fiblall2 = np.zeros((ifibm, iym2), dtype=np.float64)
        fiblall3 = np.zeros(((hwid * 2 + 1) * ifibm, iym2), dtype=np.float64)

        for ifib in range(ifibm):
            fib_start_time = time.time()
            print(f"  ファイバー {ifib + 1}/{ifibm} を処理中 (ファイル {ifile + 1}/{ifilem})")

            # 1本のファイバーの軌跡（列）を抜き出す
            fibpix = fibp[:, ifib]

            for iy2 in range(iym2):
                x_data = np.arange(b.shape[1])
                y_data = b[iy2 + iys2, :]
                interp_func = interp1d(x_data, y_data, kind='linear', bounds_error=False,fill_value=(y_data[0], y_data[-1]))

                for ix2 in range(-hwid, hwid + 1):
                    val_to_interp = fibpix[iy2] + ix2
                    fibl[ix2 + hwid, iy2] = interp_func(val_to_interp)

            # --- ガウスフィット ---
            for ix3 in range(hwid * 2 + 1):
                fibl2[ix3] = np.sum(fibl[ix3, :])

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
                arr[1] = 5.0

            for iy2 in range(iym2):
                y_data_fibl3 = fibl[:, iy2].astype(np.float32)
                x_interp_data = np.arange(fibl.shape[0])
                interp_func_fibl3 = interp1d(x_interp_data, y_data_fibl3, kind='linear', bounds_error=False,fill_value=(y_data_fibl3[0], y_data_fibl3[-1]))

                for ix2 in range(-hwid, hwid + 1):
                    fibl3[ix2 + hwid, iy2] = interp_func_fibl3(arr[1] + ix2)

                fibl4[iy2] = np.sum(fibl3[hwid - 1:hwid + 2, iy2]) - (fibl3[0, iy2] + fibl3[hwid * 2, iy2]) / 2.0 * 3.0
                fibl5[:, iy2] = fibl3[:, iy2] - (fibl3[0, iy2] + fibl3[hwid * 2, iy2]) / 2.0

            #if ifib == 45 and ifile == 0:
            #        save_path = fileF1 / 'debug_fibl3_python_fib45_lm.fit'
            #        print(f'  DEBUG: Saving fibl3 for fiber 45 to {save_path}')
            #        hdu = fits.PrimaryHDU(fibl3.T)
            #        hdu.writeto(save_path, overwrite=True)

            fiblall[(hwid * 2 + 1) * ifib: (hwid * 2 + 1) * ifib + hwid * 2 + 1, :] = fibl3
            fiblall3[(hwid * 2 + 1) * ifib: (hwid * 2 + 1) * ifib + hwid * 2 + 1, :] = fibl5
            fiblall2[ifib, :] = fibl4

            fib_end_time = time.time()
            print(f"  ファイバー {ifib + 1}/{ifibm} 処理完了。所要時間: {fib_end_time - fib_start_time:.2f}秒")

        # FITSファイルへの書き出し
        output_file_name = f'{10000 + ifile + 1:05d}_tr_python.fit'
        output_path = fileF1 / output_file_name

        # 動作実績のある、左右反転で保存
        fiblall2_flipped = fiblall2[:, ::-1]
        hdu = fits.PrimaryHDU(fiblall2_flipped)
        hdu.writeto(output_path, overwrite=True)
        file_end_time = time.time()
        print(f"ファイル {ifile + 1}/{ifilem} 処理完了。 '{output_file_name}' として保存。所要時間: {file_end_time - file_start_time:.2f}秒")

    total_end_time = time.time()
    print(f"\ntrace3.Completed. 全体所要時間: {total_end_time - total_start_time:.2f}秒")