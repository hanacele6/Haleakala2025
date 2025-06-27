import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import os
import time  # 処理時間計測のために追加

# --- 設定値 (IDLコードの変数に対応) ---
date = 'test'
fileF = Path("C:/Users/hanac/University/Senior/Mercury/")
base_dir = Path("C:/Users/hanac/University/Senior/Mercury/Haleakala2025/")
fileF1 = base_dir / f"output/{date}"
fileFd = fileF / f"data/{date}"

# --- FITSファイルの読み込み ---
# FITSファイルの読み込み前に、ファイルが存在することを確認すると良いでしょう。
# 例: if not (fileF1 / f'flat1.fit').exists(): print("Error: flat1.fit not found!"); exit()
with fits.open(os.path.join(fileF1, f'flat1.fit')) as hdul:
    flat = hdul[0].data

with fits.open(os.path.join(fileF1, f'fppoly1.fit')) as hdul:
    fibp = hdul[0].data

# fibpの形状をデバッグ表示（念のため）
print(f"DEBUG: Shape of fibp (from fppoly1.fit): {fibp.shape}")

# --- 実際のファイル名に合わせて以下の3行を修正してください ---
base_filename_prefix = 'mc01-'  # ファイル名の先頭部分
start_file_index = 1  # 連番の開始番号 (001)
end_file_index = 1  # 連番の終了番号 (004)
num_digits = 3  # 連番の桁数（例: 001, 002 のように3桁）
file_suffix = '_nhp.fits'  # ファイル名の末尾部分

file = []
for i in range(start_file_index, end_file_index + 1):
    # 連番をゼロ埋めしてファイル名を生成
    generated_filename = f'{base_filename_prefix}{i:0{num_digits}d}{file_suffix}'
    file.append(generated_filename)

# ifilem を生成されたファイルリストの長さに更新
ifilem = len(file)

print(f"Generated file list: {file}")

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
poserr_path = os.path.join(fileF1, 'poserr_python.txt')
with open(poserr_path, 'w') as lunw2:
    total_start_time = time.time()  # 全体の処理開始時間

    for ifile in range(ifilem):
        fileid = file[ifile]
        file_start_time = time.time()  # ファイルごとの処理開始時間
        print(f"\nファイル {ifile + 1}/{ifilem} を処理中: {fileid}")  # ファイルごとの進捗表示

        # --- FITSデータの読み込み ---
        data_file_path = os.path.join(fileFd, fileid)
        if not Path(data_file_path).exists():  # データファイルの存在確認
            print(f"警告: データファイル {data_file_path} が見つかりませんでした。スキップします。")
            continue  # ファイルが見つからない場合は次のループへ

        with fits.open(data_file_path) as hdul:
            b = hdul[0].data
        #with fits.open(data_file_path, do_not_scale_image_data=True) as hdul:
        #    header = hdul[0].header
        #    b_raw = hdul[0].data

        # BZEROの値を取得（存在しない場合は0.0とする）
        #bzero_val = header.get('BZERO', 0.0)

        # BZEROを足してIDLの挙動を再現
        # 計算前にfloat64に変換するのが安全
        #b = b_raw.astype(np.float64) + bzero_val
        #b = b_raw.astype(np.float64)

        #if ifile == 0:
        #    num_nans = np.sum(np.isnan(b))
        #    print(f"Python: NaNの数 in b: {num_nans}")

        # デバッグ用にBZEROの値とデータ型を表示
        #if ifile == 0:  # 最初のファイルだけでOK
        #    print(f"DEBUG: BZERO = {bzero_val}, bのデータ型: {b.dtype}")


        # 配列の初期化 (IDLのdblarrに対応)
        fibl = np.zeros((hwid * 2 + 1, iym2), dtype=np.float64)
        fibl2 = np.zeros(hwid * 2 + 1, dtype=np.float64)
        fibl3 = np.zeros_like(fibl)
        fibl4 = np.zeros(iym2, dtype=np.float64)
        fibl5 = np.zeros_like(fibl3)

        fiblall = np.zeros(((hwid * 2 + 1) * ifibm, iym2), dtype=np.float64)
        fiblall2 = np.zeros((ifibm, iym2), dtype=np.float64)
        fiblall3 = np.zeros(((hwid * 2 + 1) * ifibm, iym2), dtype=np.float64)

        for ifib in range(ifibm):
            fib_start_time = time.time()  # ファイバーごとの処理開始時間
            print(f"  ファイバー {ifib + 1}/{ifibm} を処理中 (ファイル {ifile + 1}/{ifilem})")  # ファイバーごとの進捗

            fibpix = fibp[:, ifib]

            for iy2 in range(iym2):
                x_data = np.arange(b.shape[1])
                #interp_func = interp1d(x_data, b[iy2 + iys2, :], kind='linear', fill_value="extrapolate")
                y_data = b[iy2 + iys2, :]
                interp_func = interp1d(x_data, y_data, kind='linear', bounds_error=False,fill_value=(y_data[0], y_data[-1]))

                for ix2 in range(-hwid, hwid + 1):
                    val_to_interp = fibpix[iy2] + ix2
                    #val_to_interp = np.float32(fibpix[iy2] + ix2)
                    #val_to_interp = np.round(fibpix[iy2] + np.float32(ix2), 6)
                    fibl[ix2 + hwid, iy2] = interp_func(val_to_interp)
                    #fibl[ix2 + hwid, iy2] = int(interp_func(val_to_interp))
                    interp_val = fibl[ix2 + hwid, iy2]

                    #if ifile == 0 and ifib == 45:
                        # 保存するファイル名を指定
                    #    save_path = os.path.join(fileF1, 'debug_fibl_python_fib45.fit')

                        # 画面にメッセージを表示
                    #    print(f'DEBUG: Saving fibl for fiber 45 to {save_path}')

                        # IDLの配列の向き (1327, 11) に合わせるため、転置 (.T) して保存
                    #    hdu = fits.PrimaryHDU(fibl.T)
                    #    hdu.writeto(save_path, overwrite=True)

                    #if ifile == 0 and ifib == 0:
                    #    save_filename = os.path.join(fileF1, 'fibl_Python_test.fit')
                    #    print(f'Python版のfiblを保存します: {save_filename}')
                    #    hdu = fits.PrimaryHDU(fibl.T)
                    #    hdu.writeto(save_filename, overwrite=True)

                    #DEBUG: この値をファイルに書き出す
                    #if ifile == 0 and ifib == 0: # 最初のファイル、最初のファイバーだけを比較するなど
                    #     with open("debug_interp_values3.txt", "a") as f:
                    #        f.write(f"{ifile},{ifib},{iy2},{ix2},{val_to_interp},{interp_val}\n")

            # --- ガウスフィット ---
            for ix3 in range(hwid * 2 + 1):
                fibl2[ix3] = np.sum(fibl[ix3, :])

            x_fit = np.arange(hwid * 2 + 1)

            initial_guess = [np.max(fibl2) - np.min(fibl2), float(hwid), 1.0, np.min(fibl2)]

            try:
                # maxfevを制限して無限ループを防ぐ
                params, covariance = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess, method='lm', maxfev=10000)
                #params, covariance = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess, maxfev=10000)
                arr = params

            #if ifib == 45:
            #        # ファイバー45番の場合、フィット計算をせずIDLの結果を直接使う
            #    print("  DEBUG (Fiber 45): Bypassing fit. Using hard-coded arr from IDL's result.")

                    # 注意: この値は、FITS読み込みを修正した後の最新のIDLのデバッグ出力に合わせてください。
            #    arr = np.array([67273.9, 5.08139, 1.62656, -24920.2])
            #else:
                    # その他のファイバーの場合、通常通りcurve_fitを実行する
            #    try:
                        # maxfevを制限して無限ループを防ぐ
            #        params, covariance = curve_fit(gaussian, x_fit, fibl2, p0=initial_guess, maxfev=10000)
            #        arr = params

            except RuntimeError as e:
                print(
                    f"  警告: ガウスフィットが失敗しました (ファイル {ifile}, ファイバー {ifib}): {e}. デフォルト値を使用します。")
                arr = initial_guess

            #if ifib == 44 or ifib == 45:
            #    print(
            #        f"PYTHON DEBUG: ifile={ifile}, ifib={ifib}, arr[1]={arr[1]:.6f}, abs_diff={abs(arr[1] - hwid):.6f}")
            #if ifile == 0 and ifib == 45:
            #    save_path = os.path.join(fileF1, 'debug_fibl_python.fit')
            #    print(f'  DEBUG: Saving fibl for file 0, fiber 45 to {save_path}')
            #    Tfibl = fibl.T
                #Tfibl = fibl[:, ::-1]
            #    hdu = fits.PrimaryHDU(Tfibl)
            #    hdu.writeto(save_path, overwrite=True)
            #if ifile == 0:
                # 'a'モード（追記）でファイルを開く
            #    with open(os.path.join(fileF1, 'gaussfit_params_Python.txt'), 'a') as f_gauss:
                    # ifib, amplitude, mean, stddev, offset
            #        f_gauss.write(f"{ifib},{arr[0]},{arr[1]},{arr[2]},{arr[3]}\n")


            if abs(arr[1] - hwid) > 1.5:
                lunw2.write(f"{ifile} {ifib} {arr[1]:.6f}\n")
                arr[1] = 5.0

            for iy2 in range(iym2):
                #interp_func_fibl3 = interp1d(x_interp_data, fibl[:, iy2], kind='linear', fill_value="extrapolate")
                y_data_fibl3 = fibl[:, iy2].astype(np.float32)
                x_interp_data = np.arange(fibl.shape[0])
                interp_func_fibl3 = interp1d(x_interp_data, y_data_fibl3, kind='linear', bounds_error=False,fill_value=(y_data_fibl3[0], y_data_fibl3[-1]))

                for ix2 in range(-hwid, hwid + 1):
                    fibl3[ix2 + hwid, iy2] = interp_func_fibl3(arr[1] + ix2)
                    #fibl3[ix2 + hwid, iy2] = int(interp_func_fibl3(arr[1] + ix2))
                    #val_to_interp_2nd = np.float32(arr[1] + ix2)
                    #val_to_interp_2nd = np.round(arr[1] + np.float32(ix2), 6)
                    #fibl3[ix2 + hwid, iy2] = int(interp_func_fibl3(val_to_interp_2nd))
                fibl4[iy2] = np.sum(fibl3[hwid - 1:hwid + 2, iy2]) - (fibl3[0, iy2] + fibl3[hwid * 2, iy2]) / 2.0 * 3.0
                fibl5[:, iy2] = fibl3[:, iy2] - (fibl3[0, iy2] + fibl3[hwid * 2, iy2]) / 2.0

            #if ifib == 45 and ifile == 0:
            #        save_path = os.path.join(fileF1, 'debug_fibl3_python_fib45_lm.fit')
            #        print(f'  DEBUG: Saving fibl3 for fiber 45 to {save_path}')
            #        # IDLのshape (1327, 11) に合わせるため転置(.T)する
            #        hdu = fits.PrimaryHDU(fibl3.T)
            #        hdu.writeto(save_path, overwrite=True)

            fiblall[(hwid * 2 + 1) * ifib: (hwid * 2 + 1) * ifib + hwid * 2 + 1, :] = fibl3
            fiblall3[(hwid * 2 + 1) * ifib: (hwid * 2 + 1) * ifib + hwid * 2 + 1, :] = fibl5
            fiblall2[ifib, :] = fibl4

            fib_end_time = time.time()
            print(f"  ファイバー {ifib + 1}/{ifibm} 処理完了。所要時間: {fib_end_time - fib_start_time:.2f}秒")

        output_file_name = f'{10000 + ifile + 1:02d}_tr_python.fit'
        output_path = os.path.join(fileF1, output_file_name)

        #hdu = fits.PrimaryHDU(fiblall2.T)
        fiblall2_flipped = fiblall2[:, ::-1]
        hdu = fits.PrimaryHDU(fiblall2_flipped)
        hdu.writeto(output_path, overwrite=True)
        file_end_time = time.time()
        print(f"ファイル {ifile + 1}/{ifilem} 処理完了。所要時間: {file_end_time - file_start_time:.2f}秒")

    total_end_time = time.time()
    print(f"\ntrace3.Completed. 全体所要時間: {total_end_time - total_start_time:.2f}秒")